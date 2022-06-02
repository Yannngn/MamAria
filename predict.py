import warnings
import albumentations as A
import os
import torch
import torch.nn as nn
import wandb

from albumentations.pytorch import ToTensorV2
from datetime import datetime
from munch import munchify, unmunchify
from tqdm import tqdm
from yaml import safe_load

from model import UNET
from utils.save_images import save_test_as_imgs
from loggers.logs import log_submission
from losses.loss import TverskyLoss
from utils.utils import get_loaders, get_weights, load_checkpoint

os.environ['WANDB_MODE'] = 'offline'
BEGIN = datetime.now().strftime("%Y%m%d_%H%M%S")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))
PATH = os.path.dirname(__file__)

warnings.filterwarnings("ignore")

def predict_fn(test_loader, model, loss_fn, time):
    loop = tqdm(test_loader)

    model.eval()

    save_test_as_imgs(test_loader, time=BEGIN)

    for _, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.long().to(DEVICE)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
    log_submission(test_loader, model, loss.item(), time=time)
    
    model.train()

    return loss.item()

def main():
    wandb.init(
        project = CONFIG.PROJECT.PROJECT_NAME,
        entity = CONFIG.PROJECT.PROJECT_TEAM,
        config = unmunchify(CONFIG.HYPERPARAMETERS))

    config = wandb.config

    train_transforms = A.Compose([ToTensorV2(),],)
    val_transforms = A.Compose([ToTensorV2(),],)

    test_transforms = A.Compose([ToTensorV2(),],)
    _, _, test_loader = get_loaders(CONFIG.PATHS.TRAIN_IMG_DIR,
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

    if config.WEIGHTS is not None:
        weights = get_weights(CONFIG.PATHS.TRAIN_MASK_DIR, 
                              CONFIG.IMAGE.MASK_LABELS, 
                              multiplier=config.MULTIPLIER,
                              device=DEVICE)

    else:
        weights = None
    
    if config.LOSS_FUNCTION == "crossentropy":
        loss_fn = nn.CrossEntropyLoss(weight = weights)
    elif config.LOSS_FUNCTION == "tversky":
        loss_fn = TverskyLoss(alpha=0.5, beta=0.5, weight=weights)
    else:
        raise KeyError(f"loss function {config.LOSS_FUNCTION} not recognized.")

    load_checkpoint(torch.load("data/checkpoints/20220216_212306_best_checkpoint.pth.tar", map_location=torch.device(DEVICE)), model, optimizer=None, scheduler=None)

    predict_fn(
        test_loader,
        model,
        loss_fn,
        time = BEGIN
    )
    
    wandb.finish()  

if __name__ == "__main__":
    main()