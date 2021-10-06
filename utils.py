import torch
import numpy as np
import torchvision
import torch.nn as nn
from dataset_v1 import PhantomDataset
from torch.utils.data import DataLoader
from datetime import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):

    train_ds = PhantomDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = PhantomDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device=DEVICE):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y = (y * 3).unsqueeze(1)

            preds = torch.softmax(model(x), 1)
            preds_labels = torch.argmax(preds, 1).unsqueeze(1)

            num_correct += (preds_labels == y).sum()
            num_pixels += torch.numel(preds_labels)

            dice_score = dice_coef_multilabel(y, preds_labels, 4)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(loader, model, epoch=None, folder="data/predictions/", device=DEVICE):
    print("=> Saving predictions as images")
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.softmax(model(x), 1)
            preds_labels = torch.argmax(preds, 1,)
            preds_labels = label_to_pixel(preds_labels)

            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            torchvision.utils.save_image(preds_labels, f"{folder}{now}pred_e{epoch}_i{idx}.png",)
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{now}val_e{epoch}_i{idx}.png")

    model.train()

def label_to_pixel(preds):
    # preds[preds == 1] = 1/3
    # preds[preds == 3] = 1.0
    # preds[preds == 2] = 2/3
    # preds[preds == 0] = 0.0

    preds = preds / 3

    return preds.unsqueeze(1).float()

def dice_coef(y_true, y_pred):

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels = 4):
    dice=0
    y_true, y_pred = y_true.numpy(), y_pred.numpy()
    for index in range(numLabels):
        dice += dice_coef(y_true == index, y_pred == index)
    return dice / numLabels # taking average
