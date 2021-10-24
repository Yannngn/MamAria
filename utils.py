import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from cv2 import imread
from sklearn.metrics import precision_score, recall_score, f1_score

import os
import numpy as np
from datetime import datetime

from dataset import PhantomDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREDICTIONS_DIR = "data/predictions/"

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer, scheduler):
    print("=> Loading checkpoint")
    try:
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
    except KeyError:
        pass
    try:
        model.load_state_dict(checkpoint)
    except KeyError as e:
        raise ValueError(f'Key {e} is different from expected "state_dict"')

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

def get_weights(mask_dir, num_labels, device=DEVICE, multiplier = [1, 1, 1, 1]):
    weights = np.zeros(num_labels)
    multiplier = np.array(multiplier)
    total_pixels = 0
    mask_files = [os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if file.endswith('.png')]
    
    for mask in mask_files:
        mask = imread(mask)
    
        if total_pixels == 0:
            total_pixels = mask.shape[1] * mask.shape[2]

        temp = []
        
        for i in range(num_labels):
            temp.append((mask == i).sum())
        
        weights += temp
        den = weights / (total_pixels * len(mask))
        out = np.divide(multiplier, den, out = np.zeros_like(multiplier, dtype = float), where = den!=0)

    return torch.tensor(out).float().to(device)

def check_accuracy(loader, model, num_labels, device=DEVICE):
    num_correct = 0
    num_pixels = 0
    
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x)
            y = y.to(device)

            y = (y).unsqueeze(1)

            preds_labels = torch.argmax(preds, 1).unsqueeze(1)
          
            num_correct += (preds_labels == y).sum()
            num_pixels += torch.numel(preds_labels)
            
            y = y.to('cpu').numpy()
            preds_labels = preds_labels.to('cpu').numpy()

            acc = evaluate_segmentation(preds_labels, y, num_labels, score_averaging = None)

    model.train()

    return [num_correct, num_pixels, acc]

def print_and_save_results(n0, n1, lst, trainl, vall, time, folder=PREDICTIONS_DIR):
    lst.append(trainl)
    lst.append(vall)
    print(f"Got {n0}/{n1} with Global Accuracy: {lst[0] * 100:.4f}%",
          f"\nClasses Accuracy: {lst[1]}",
          f"\nRecall: {lst[3]}",
          f"\nTrain Loss: {lst[6]}",
          f"\nVal Loss: {lst[7]}")
    
    with open(folder+f'{time}_preds.csv','a') as fd:
        fd.write(';'.join(map(str, [l for l in lst])) + '\n')

def save_predictions_as_imgs(loader, model, epoch, folder=PREDICTIONS_DIR, device=DEVICE):
    print("=> Saving predictions as images")
    model.eval()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    with torch.no_grad():    
        for idx, (x, _) in enumerate(loader):
            x = x.to(device)
            preds_labels = torch.argmax(model(x), 1)
            preds_labels = label_to_pixel(preds_labels)

            save_image(preds_labels, f"{folder}{now}_pred_e{epoch}_i{idx}.png")
            
    model.train()

def save_validation_as_imgs(loader, folder=PREDICTIONS_DIR, device=DEVICE):
    print("=> Saving predictions as images")

    for idx, (_, y) in enumerate(loader):
        y = y.to(device)
        val = (y / y.max()).unsqueeze(1)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_image(val, f"{folder}{now}_val_i{idx}.png")
        
def label_to_pixel(preds, col='l'):
    if col == 'l':
        preds = preds / 3
        preds = preds.unsqueeze(1).float()
        return preds

    else:
        preds = preds[:,1:,:,:]
        return preds.float()

# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Replace the nans with a 0.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(0.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies

def compute_mean_iou(pred, label, mean=False):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I, U = np.zeros(num_unique_labels), np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val
        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))
    
    out = I/U
    
    if mean:
        out = np.mean(out)

    return out

def evaluate_segmentation(pred, label, num_classes, score_averaging=None):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging, zero_division = 0)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging, zero_division = 0)

    iou = compute_mean_iou(flat_pred, flat_label)

    return [global_accuracy, np.array(class_accuracies), np.array(prec), np.array(rec), np.array(f1), np.array(iou)]