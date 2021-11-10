import os
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from datetime import datetime
from munch import munchify, unmunchify
from yaml import safe_load

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import PhantomDataset
from utils import get_weights, load_checkpoint, log_submission, save_validation_as_imgs
from model import UNET

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))
PATH = os.path.dirname(__file__)
BEGIN = datetime.now().strftime("%Y%m%d_%H%M%S")

def predict_fn(test_loader, model, loss_fn, time):
    loop = tqdm(test_loader)
    
    epoch = load_checkpoint(torch.load(PATH+f"/data/checkpoints/{time}_best_checkpoint.pth.tar"), model, optimizer=None, scheduler=None)

    model.eval()
    
    for idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.long().to(DEVICE)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
    log_submission(test_loader, model, loss.item(), epoch, idx, time=time)

    model.train()
        
    wandb.finish()
    return loss.item()

def get_loader(
    test_dir,
    test_maskdir,
    batch_size,
    test_transform,
    num_workers=4,
    pin_memory=True,
):

    test_ds = PhantomDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
        
    return test_loader


def main():
    wandb.init(
        project = CONFIG.PROJECT.PROJECT_NAME,
        entity = CONFIG.PROJECT.PROJECT_TEAM,
        config = unmunchify(CONFIG.HYPERPARAMETERS))

    config = wandb.config

    test_transforms = A.Compose([ToTensorV2(),],)

    test_loader = get_loader(
                        CONFIG.PATHS.TEST_IMG_DIR,
                        CONFIG.PATHS.TEST_MASK_DIR,
                        CONFIG.HYPERPARAMETERS.BATCH_SIZE,
                        test_transforms,
                        CONFIG.PROJECT.NUM_WORKERS,
                        CONFIG.PROJECT.PIN_MEMORY,
                    )

    model = UNET(in_channels = CONFIG.IMAGE.IMAGE_CHANNELS, classes = CONFIG.IMAGE.MASK_LABELS, config = config).to(DEVICE)
    model = nn.DataParallel(model)

    if CONFIG.HYPERPARAMETERS.WEIGHTS:
        weights = get_weights(CONFIG.PATHS.TRAIN_MASK_DIR, CONFIG.IMAGE.MASK_LABELS, DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight = weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    save_validation_as_imgs(test_loader, time=BEGIN, folder = CONFIG.PATHS.SUBMISSIONS_DIR, device = DEVICE)

    predict_fn(
        test_loader,
        model,
        loss_fn,
        time = BEGIN
    )
        
    wandb.finish()

if __name__ == "__main__":
    main()