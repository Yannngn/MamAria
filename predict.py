import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from datetime import datetime

import torch.nn as nn
from torch.utils.data import DataLoader
from main import BATCH_SIZE

from model import UNET
from dataset import PhantomDataset
from utils import (
    load_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
    save_validation_as_imgs,
    get_weights,
    print_and_save_results
)

# Hyperparameters etc.
LEARNING_RATE = 3e-4 ### Begin with 3e-4, 96.41% now 8e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 25
NUM_WORKERS = 12
IMAGE_HEIGHT = 512  # 256 originally
IMAGE_WIDTH = 301  # 98 originally
IMAGE_CHANNELS = 1
MASK_CHANNELS = 1
MASK_LABELS = 4
PIN_MEMORY = True
CHECKPOINT_PATH = "checkpoints/20211020_191703_best_checkpoint.pth.tar"
PARENT_DIR = os.path.abspath(__file__)
INPUT_IMG_DIR = "data/submission/input/phantom/"
INPUT_MASK_DIR = "data/submission/input/mask/"
PREDICTIONS_DIR = "data/submission/predictions/"

torch.manual_seed(19)

def predict_fn(loader, model, loss_fn):
    loop = tqdm(loader)
    model.eval()
    
    for _, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.long().to(DEVICE)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    
    return loss.item()

def main():

    pred_transforms = A.Compose([ToTensorV2(),],)

    pred_loader = get_loader(
        INPUT_IMG_DIR,
        INPUT_MASK_DIR,
        BATCH_SIZE,
        pred_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    model = UNET(in_channels = IMAGE_CHANNELS, classes= MASK_LABELS).to(DEVICE)
    weights = get_weights(INPUT_MASK_DIR, MASK_LABELS, DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight = weights)

    BEGIN = datetime.now().strftime("%Y%m%d_%H%M%S")

    load_checkpoint(torch.load(CHECKPOINT_PATH), model, optimizer=None, scheduler=None)

    save_validation_as_imgs(pred_loader, folder = PREDICTIONS_DIR, device = DEVICE)
    
    accuracies = check_accuracy(pred_loader, model, MASK_LABELS, DEVICE)

    pred_loss = predict_fn(pred_loader, model, loss_fn)

    print_and_save_results(accuracies[0], accuracies[1], accuracies[2], 0, pred_loss, BEGIN)

    save_predictions_as_imgs(pred_loader, model, 0, folder = PREDICTIONS_DIR, device = DEVICE)

def get_loader(
    pred_dir,
    pred_maskdir,
    batch_size,
    pred_transform,
    num_workers=4,
    pin_memory=True,
):

    pred_ds = PhantomDataset(
        image_dir=pred_dir,
        mask_dir=pred_maskdir,
        transform=pred_transform,
    )

    pred_loader = DataLoader(
        pred_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return pred_loader

if __name__ == "__main__":
    main()