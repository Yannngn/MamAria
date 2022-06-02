import cv2
import numpy as np
import torch
import os

from munch import munchify
from torch.utils.data import DataLoader
from tqdm import tqdm
from yaml import safe_load

from datasets.dataset import PhantomDataset
from utils.metrics import evaluate_segmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))

def check_accuracy(loader, model, device=DEVICE):
    loop = tqdm(loader)

    model.eval()
    with torch.no_grad():
        for x, y in loop:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)

            dict_eval = evaluate_segmentation(preds, y, score_averaging = None)
    
    model.train()

    return dict_eval

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print(f'='.center(125, '='))
    print("Saving checkpoint ...")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer, scheduler):
    print(f'='.center(125, '='))
    print("Loading checkpoint ...")
    try:
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        return
    except KeyError:
        pass
    try:
        model.load_state_dict(checkpoint)
    except KeyError as e:
        raise ValueError(f'Key {e} is different from expected "state_dict"')
    return checkpoint["epoch"]

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, test_dir, test_maskdir, batch_size, 
                train_transform, val_transform, test_transform, num_workers=4, pin_memory=True):

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

    if test_dir is not None:
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
        
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader

def get_weights(mask_dir, num_labels, device=DEVICE, multiplier = CONFIG.HYPERPARAMETERS.MULTIPLIER):
    weights = np.zeros(num_labels)
    multiplier = np.array(multiplier)
    total_pixels = 0
    mask_files = [os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if file.endswith('.png')]
    
    for mask in mask_files:
        temp = []
        mask = cv2.imread(mask)
    
        if total_pixels == 0: total_pixels = mask.shape[1] * mask.shape[2]

        for i in range(num_labels): temp.append((mask == i).sum())
        
        weights += temp
    
    den = weights / (total_pixels * len(mask))
    out = np.divide(multiplier, den, out = np.zeros_like(multiplier, dtype = float), where = den!=0)
    
    return torch.tensor(out).float().to(device)
