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

def get_weights(mask_dir, num_labels, device=DEVICE, multiplier = [.5, 1, 2, 2]):
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
        
        out = multiplier / (weights / (total_pixels * len(mask)))
    
    print(out)

    return torch.tensor(out).float().to(device)

def check_accuracy(loader, model, num_labels, device=DEVICE):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y = (y * 3.).unsqueeze(1)
            
            # print("X to device:", x.shape, x.max(), x.min(), x.mean())
            # print("Y unsqueeze:", y.shape, y.max(), y.min(), y.mean())

            preds = torch.log_softmax(model(x), 1)
            
            # print("Preds:", preds.shape, preds.max(), preds.min(), preds.mean())
            
            preds_labels = torch.argmax(preds, 1).unsqueeze(1)
            
            num_correct += (preds_labels == y).sum()
            num_pixels += torch.numel(preds_labels)
            #print("Preds Labels:", preds_labels.shape, preds_labels.max(), preds_labels.min(), preds_labels.unique(), preds_labels.float().mean())    

            dice_score = dice_loss(y, preds_labels, num_labels)
    
    acc = evaluate_segmentation(preds_labels, y, 4, score_averaging='weighted')
    print(f"Got {num_correct}/{num_pixels} with Global Accuracy: {acc[0] * 100:.2f}",
          f"\nClasses Accuracy: {acc[1]}",
          f"\nPrecisão: {acc[2]}",
          f"\nRecall: {acc[3]}",
          f"\nF1: {acc[4]}",
          f"\nDice loss: {dice_score/len(loader)}",
          f"\nMean IoU: {acc[5]}")
    
    model.train()

def save_predictions_as_imgs(loader, model, epoch, folder="data/predictions/", device=DEVICE):
    print("=> Saving predictions as images")
    model.eval()
    with torch.no_grad():    
        for idx, (x, _) in enumerate(loader):
            x = x.to(device)

            preds = torch.log_softmax(model(x), 1)
            preds_labels = torch.argmax(preds, 1)
            
            preds_labels = label_to_pixel(preds_labels)

            now = datetime.now().strftime("%Y%m%d_%H%M%S")

            save_image(preds_labels, f"{folder}{now}_pred_e{epoch}_i{idx}.png")

            # preds_labels = ToPILImage()(preds_labels).convert("L")
            # preds_labels.save(f"{folder}{now}_pred_e{epoch}_i{idx}.png")
            
    model.train()

def save_validation_as_imgs(loader, folder="data/predictions/", device=DEVICE):
    print("=> Saving predictions as images")

    for idx, (_, y) in enumerate(loader):
        y = y.to(device)
        val = y.unsqueeze(1)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_image(val, f"{folder}{now}_val_i{idx}.png")
        #val = ToPILImage()(y).convert("L")
        #val.save(f"{folder}{now}_val_i{idx}.png")
        
def label_to_pixel(preds):
    preds = preds / 3

    return preds.unsqueeze(1).float()

def dice_coef(y_true, y_pred):

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, num_labels):
    dice=0
    y_true, y_pred = y_true.numpy(), y_pred.numpy()
    for index in range(num_labels):
        dice += dice_coef(y_true == index, y_pred == index)
    return dice / num_labels # taking average

def dice_loss(y_true, y_pred, num_labels):
    loss = 1 - dice_coef_multilabel(y_true, y_pred, num_labels)
    return loss

# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    pred, label = pred.numpy(), label.numpy()
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    pred, label = pred.numpy(), label.numpy()
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies

def compute_mean_iou(pred, label):
    pred, label = pred.numpy(), label.numpy()
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        
        pred_i = pred == val
        label_i = label == val
        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    return np.mean(I / U)

def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou