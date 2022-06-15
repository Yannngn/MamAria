import wandb
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import warnings

print(torchmetrics.__version__)

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datetime import datetime

from munch import munchify, unmunchify
from yaml import safe_load

from models.unet import UNET
from train import train_loop
from utils.early_stopping import EarlyStopping
from utils.save_images import save_validation_as_imgs, save_ellipse_validation_as_imgs
from losses.loss import TverskyLoss
from utils.utils import load_checkpoint, get_loaders, get_weights

os.environ['WANDB_MODE'] = 'offline'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARENT_DIR = os.path.abspath(__file__)
BEGIN = datetime.now().strftime("%Y%m%d_%H%M%S")

with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))

warnings.filterwarnings("ignore")

def main():
    wandb.init(
        project = CONFIG.PROJECT.PROJECT_NAME,
        entity = CONFIG.PROJECT.PROJECT_TEAM,
        config = unmunchify(CONFIG.HYPERPARAMETERS))

    config = wandb.config

    train_transforms = A.Compose([ToTensorV2(),],)
    val_transforms = A.Compose([ToTensorV2(),],)
    test_transforms = A.Compose([ToTensorV2(),],)
    
    global_metrics = [
        torchmetrics.Accuracy(num_classes=CONFIG.IMAGE.MASK_LABELS).to(DEVICE),
        torchmetrics.F1(num_classes=CONFIG.IMAGE.MASK_LABELS, mdmc_average='global').to(DEVICE),
        torchmetrics.Precision(num_classes=CONFIG.IMAGE.MASK_LABELS, mdmc_average='global').to(DEVICE),
        torchmetrics.Recall(num_classes=CONFIG.IMAGE.MASK_LABELS, mdmc_average='global').to(DEVICE),
        torchmetrics.Specificity(num_classes=CONFIG.IMAGE.MASK_LABELS, mdmc_average='global').to(DEVICE),
    ]

    global_metrics_names = ["accuracy", "f1", "precision", "recall", "specificity"]

    label_metrics = [
        torchmetrics.Accuracy(num_classes=CONFIG.IMAGE.MASK_LABELS, average=None, mdmc_average='global').to(DEVICE),
        torchmetrics.F1(num_classes=CONFIG.IMAGE.MASK_LABELS, average=None, mdmc_average='global').to(DEVICE),
        torchmetrics.Precision(num_classes=CONFIG.IMAGE.MASK_LABELS, average=None, mdmc_average='global').to(DEVICE),
        torchmetrics.Recall(num_classes=CONFIG.IMAGE.MASK_LABELS, average=None, mdmc_average='global').to(DEVICE),
        torchmetrics.Specificity(num_classes=CONFIG.IMAGE.MASK_LABELS, average=None, mdmc_average='global').to(DEVICE),
    ]
      
    label_metrics_names = ["accuracy", "f1", "precision", "recall", "specificity"]
    
    global_metrics = dict(zip(global_metrics_names, global_metrics))
    label_metrics = dict(zip(label_metrics_names, label_metrics))
    
    train_loader, val_loader, _ = get_loaders(CONFIG.PATHS.TRAIN_IMG_DIR,
                                                        CONFIG.PATHS.TRAIN_MASK_DIR,
                                                        CONFIG.PATHS.VAL_IMG_DIR,
                                                        CONFIG.PATHS.VAL_MASK_DIR,
                                                        CONFIG.PATHS.TEST_IMG_DIR,
                                                        CONFIG.PATHS.TEST_MASK_DIR,
                                                        config.BATCH_SIZE,
                                                        train_transforms,
                                                        val_transforms,
                                                        test_transforms,
                                                        0,
                                                        CONFIG.PROJECT.PIN_MEMORY,
                                                        )

    model = UNET(in_channels = CONFIG.IMAGE.IMAGE_CHANNELS, classes = CONFIG.IMAGE.MASK_LABELS, config = config).to(DEVICE)
    model = nn.DataParallel(model)

    if config.WEIGHTS is not None:
        weights = get_weights(CONFIG.PATHS.TRAIN_MASK_DIR, 
                              CONFIG.IMAGE.MASK_LABELS, 
                              multiplier=config.MULTIPLIER,
                              device=DEVICE)

    else: weights = None
    
    if config.LOSS_FUNCTION == "crossentropy": loss_fn = nn.CrossEntropyLoss(weight = weights)
    elif config.LOSS_FUNCTION == "tversky": loss_fn = TverskyLoss(alpha=0.5, beta=0.5, weight=weights)
    else: raise KeyError(f"loss function {config.LOSS_FUNCTION} not recognized.")
    
    if config.OPTIMIZER == 'adam': 
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE
            )
        
    elif config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                              lr=config.LEARNING_RATE, 
                              momentum=0.9, 
                              nesterov=True, 
                              weight_decay=0.0001
                              )
    else: raise KeyError(f"optimizer {config.OPTIMIZER} not recognized.")
    
    if CONFIG.PROJECT.SCHEDULER: 
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    load_epoch = 0
    if CONFIG.PROJECT.LOAD_MODEL: 
        load_epoch = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer, scheduler)

    scaler = torch.cuda.amp.GradScaler()

    stopping = EarlyStopping(patience=10, wait=30)
    
    save_validation_as_imgs(val_loader, time = BEGIN)
    save_ellipse_validation_as_imgs(val_loader, time = BEGIN)

    train_loop(
        train_loader, 
        val_loader, 
        model, 
        optimizer, 
        scheduler, 
        loss_fn, 
        scaler, 
        stopping,
        global_metrics, 
        label_metrics,
        config,
        load_epoch,
        time = BEGIN
    )

if __name__ == "__main__":
    main()

