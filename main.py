import wandb
import os
import torch
import torch.nn as nn
import torchgeometry as tgm
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datetime import datetime

from munch import munchify, unmunchify
from yaml import safe_load

from model import UNET
from early_stopping import EarlyStopping
from utils import load_checkpoint, get_loaders, save_validation_as_imgs, get_weights
from train import train_loop
from predict import predict_fn
from loss import TverskyLoss
from summary import summary
import pytorch_model_summary as pms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARENT_DIR = os.path.abspath(__file__)
BEGIN = datetime.now().strftime("%Y%m%d_%H%M%S")

with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))

def main():
    wandb.init(
        project = CONFIG.PROJECT.PROJECT_NAME,
        entity = CONFIG.PROJECT.PROJECT_TEAM,
        config = unmunchify(CONFIG.HYPERPARAMETERS))

    config = wandb.config

    train_transforms = A.Compose([ToTensorV2(),],)
    val_transforms = A.Compose([ToTensorV2(),],)

    test_transforms = A.Compose([ToTensorV2(),],)
    train_loader, val_loader, test_loader = get_loaders(CONFIG.PATHS.TRAIN_IMG_DIR,
                                                        CONFIG.PATHS.TRAIN_MASK_DIR,
                                                        CONFIG.PATHS.VAL_IMG_DIR,
                                                        CONFIG.PATHS.VAL_MASK_DIR,
                                                        CONFIG.PATHS.TEST_IMG_DIR,
                                                        CONFIG.PATHS.TEST_MASK_DIR,
                                                        config.BATCH_SIZE,
                                                        train_transforms,
                                                        val_transforms,
                                                        test_transforms,
                                                        2,
                                                        CONFIG.PROJECT.PIN_MEMORY,
                                                        )

    model = UNET(in_channels = CONFIG.IMAGE.IMAGE_CHANNELS, classes = CONFIG.IMAGE.MASK_LABELS, config = config).to(DEVICE)
    model = nn.DataParallel(model)
    #print(model)


    if config.WEIGHTS is not None:
        weights = get_weights(CONFIG.PATHS.TRAIN_MASK_DIR, 
                              CONFIG.IMAGE.MASK_LABELS, 
                              multiplier=config.MULTIPLIER,
                              device=DEVICE
                              )

    else:
        weights = None
    
    if config.LOSS_FUNCTION == "crossentropy":
        loss_fn = nn.CrossEntropyLoss(weight = weights)
    elif config.LOSS_FUNCTION == "tversky":
        loss_fn = TverskyLoss(alpha=0.5, beta=0.5, weights=weights)
    else:
        raise KeyError(f"loss function {config.LOSS_FUNCTION} not recognized.")
    
    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                              lr=config.LEARNING_RATE, 
                              momentum=0.9, 
                              nesterov=True, 
                              weight_decay=0.0001
                              )
    else:
        raise KeyError(f"optimizer {config.OPTIMIZER} not recognized.")
    
    if CONFIG.PROJECT.SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    load_epoch = 0
    if CONFIG.PROJECT.LOAD_MODEL:
        load_epoch = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer, scheduler)

    scaler = torch.cuda.amp.GradScaler()

    stopping = EarlyStopping(patience=15, wait=50)
    
    save_validation_as_imgs(val_loader, time = BEGIN)

    train_loop(
        train_loader, 
        val_loader, 
        model, 
        optimizer, 
        scheduler, 
        loss_fn, 
        scaler, 
        stopping,
        config,
        load_epoch,
        time = BEGIN
    )

    save_validation_as_imgs(test_loader, time=BEGIN)

    predict_fn(
        test_loader,
        model,
        loss_fn,
        time = BEGIN
    )

if __name__ == "__main__":
    main()