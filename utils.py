import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from cv2 import imread
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from torchmetrics import AUROC

import os
import csv
import numpy as np
from yaml import safe_load
from munch import munchify
import wandb
from datetime import datetime

from dataset import PhantomDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer, scheduler):
    print("=> Loading checkpoint")
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

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    test_dir,
    test_maskdir,
    batch_size,
    train_transform,
    val_transform,
    test_transform,
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
        mask = imread(mask)
    
        if total_pixels == 0:
            total_pixels = mask.shape[1] * mask.shape[2]

        temp = []
        
        for i in range(num_labels):
            temp.append((mask == i).sum())
        
        weights += temp
    den = weights / (total_pixels * len(mask))
    out = np.divide(multiplier, den, out = np.zeros_like(multiplier, dtype = float), where = den!=0)
    print(out)
    return torch.tensor(out).float().to(device)

def check_accuracy(loader, model, device=DEVICE):
    num_correct = 0
    num_pixels = 0
    
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x) #([14, 3, 512, 301])
            pred_long = preds.long
            
            y = y.to(device)

            y_long = y.long() #B, H, W
            
            y = (y).unsqueeze(1)

            preds_labels = torch.argmax(preds, 1).unsqueeze(1) #(14, 1, 512, 301) -> flatten (2157568,)

            probs_labels = F.softmax(preds, 1) #(14, 3, 512, 301) acho q tá fazendo a mais aqui
                
            num_correct += (preds_labels == y).sum()
            num_pixels += torch.numel(preds_labels)
            
            y = y.cpu().detach().numpy()
            #bin_y = bin_y.cpu().detach().numpy()

            #pred_roc_labels = preds_labels

            preds_labels = preds_labels.cpu().detach().numpy()

            #probs_labels =probs_labels.cpu().detach().numpy()

            dict_eval = evaluate_segmentation(preds_labels, y, score_averaging = None)

            roc_score = roc(probs_labels, y_long)

    model.train()

    return num_correct, num_pixels, dict_eval, roc_score

def log_predictions(
    val_loader, 
    model,
    loss_train, 
    loss_val, 
    epoch, 
    time=0, 
    folder=CONFIG.PATHS.PREDICTIONS_DIR,
    device=DEVICE
):
    num_correct, num_pixels, dict_eval, roc_score  = check_accuracy(val_loader, model, device)

    dict_eval['loss_train'] = loss_train
    dict_eval['loss_val'] = loss_val

    roc_score['loss_train'] = loss_train
    roc_score['loss_val'] = loss_val

    for key in roc_score:
        print (key,':', roc_score[key])

    print(f"Got {num_correct} of {num_pixels} pixels;")
    for key in dict_eval:
        print (key,':', dict_eval[key])

    with open(folder+f'{time}_preds.csv','a') as f:
        w = csv.DictWriter(f, dict_eval.keys())
        w.writeheader()
        w.writerow(dict_eval)
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_predictions_as_imgs(val_loader, model, epoch, dict_eval, time=now, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=device)

def log_submission(loader, model, loss_test, time=0, folder=CONFIG.PATHS.PREDICTIONS_DIR, device = DEVICE):
    num_correct, num_pixels, dict_subm = check_accuracy(loader, model, device)
    dict_subm['loss_subm'] = loss_test

    print(f"Got {num_correct} of {num_pixels} pixels;")
    for key in dict_subm:
        print (key,':', dict_subm[key])

    with open(folder+f'{time}_submission.csv','a') as f:
        w = csv.DictWriter(f, dict_subm.keys())
        w.writeheader()
        w.writerow(dict_subm)
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_submission_as_imgs(loader, model, dict_subm, time=now, folder=CONFIG.PATHS.SUBMISSIONS_DIR, device=device)

def save_predictions_as_imgs(loader, model, epoch, dict_eval, folder=CONFIG.PATHS.PREDICTIONS_DIR, time=0, device=DEVICE):
    print("=> Saving predictions as images ...")
    
    model.eval()
    with torch.no_grad():    
        for idx, (x, _) in enumerate(loader):
            x = x.to(device)
            preds_labels = torch.argmax(model(x), 1)
            #print(preds_labels.unique(), "pred labels")
            preds_labels = label_to_pixel(preds_labels)
            #print(preds_labels.unique(), "pred labels to pixel")
            img = folder + f"{time}_pred_e{epoch}_i{idx}.png"
            save_image(preds_labels, img)
            dict_eval[f'prediction_i{idx}'] = wandb.Image(img)
            
            
    model.train()
    wandb.log(dict_eval)

def save_submission_as_imgs(loader, model, dict_subm, folder=CONFIG.PATHS.SUBMISSIONS_DIR, time=0, device=DEVICE):
    print("=> Saving submission images ...")
    
    model.eval()
    with torch.no_grad():    
        for idx, (x, _) in enumerate(loader):
            x = x.to(device)
            preds_labels = torch.argmax(model(x), 1)
            preds_labels = label_to_pixel(preds_labels)
            img = folder + f"{time}_submission_i{idx}.png"
            save_image(preds_labels, img)
            dict_subm[f'submission_i{idx}'] = wandb.Image(img)
            
    wandb.log(dict_subm)

def save_validation_as_imgs(loader, folder=CONFIG.PATHS.PREDICTIONS_DIR, time=0, device=DEVICE):
    print("=> Saving validation images ...")
    dict_val = {}
    for idx, (_, y) in enumerate(loader):
        y = y.to(device)
        #print(y.unique(), "y val")
        val = (y / y.max()).unsqueeze(1)
        #print(val.unique(), "val label")
        img = f"{folder}{time}_val_i{idx}.png"
        save_image(val, img)
        dict_val[f'validation_i{idx}'] = wandb.Image(img)
    
    wandb.log(dict_val)

def label_to_pixel(preds, col='l'):
    if col == 'l':
        preds = preds / (CONFIG.IMAGE.MASK_LABELS - 1) #0, 1, 2 
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

def compute_iou(pred, label, mean=False):
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

def compute_dice_score(true, logits, eps=1e-7):
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_score = (2. * intersection / (cardinality + eps)).mean()
    return dice_score

def evaluate_segmentation(pred, label, score_averaging=None):
    
    flat_pred = pred.flatten()
    flat_label = label.flatten()
    
    global_accuracy = compute_global_accuracy(flat_pred, flat_label)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging, zero_division = 0)
    iou = compute_iou(flat_pred, flat_label)
   
    #dice = compute_dice_score(flat_pred, flat_label)

    dict_eval = {"accuracy":global_accuracy,
    #"roc_auc":roc_score
    }

    for i in range(CONFIG.IMAGE.MASK_LABELS):
        dict_eval[f'accuracy_label_{i}'] = prec[i]
        dict_eval[f'recall_label_{i}'] = rec[i]
        dict_eval[f'iou_label_{i}'] = iou[i]
        #dict_eval[f'roc_score{i}'] = roc_score[i]
        #dict_eval[f'dice_label_{i}'] = dice[i]

    return dict_eval

def roc (pred ,label):
    #pred (float tensor): (N, C, ...) (multiclass) tensor with probabilities, where C is the number of classes.
    #label (long tensor): (N, ...) or (N, C, ...) with integer labels
    #if the preds and target tensor have the same size the input will be interpretated as multilabel 
    #if preds have one dimension more than the target tensor the input will be interpretated as multiclass.
    preds = pred
    labels = label
    num_classes =CONFIG.IMAGE.MASK_LABELS
    

    """roc_macro = AUROC(preds, labels, num_classes, average='macro', max_fpr=None,
                      compute_on_step=True, dist_sync_on_step=False, process_group=None, dist_sync_fn=None)

    roc_weight = AUROC (preds, labels, num_classes, average='weighted', max_fpr=None,
                        compute_on_step=True, dist_sync_on_step=False, process_group=None, dist_sync_fn=None)"""
    
    roc_per_class = AUROC (preds, labels, num_classes, average=None)

    roc_score = {#"roc_macro":roc_macro,
                 #"roc_weight":roc_weight,
                 "roc_per_class":roc_per_class,
    }   

    # Plot all ROC curves
    plt.figure()
    """plt.plot(
        roc_macro,
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_macro),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        roc_weight,
        label="macro-weight ROC curve (area = {0:0.2f})".format(roc_weight),
        color="navy",
        linestyle=":",
        linewidth=4,
    )"""

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(CONFIG.IMAGE.MASK_LABELS), colors):
        plt.plot(
            roc_per_class,
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_per_class[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()

    wandb.log({"ROC": plt})
    
    return roc_score


    

    
def ROC (y_test, y_score):

    y_test_flatten =y_test.cpu().detach().numpy().flatten() 
    
    y_test_flatten = label_binarize(y_test_flatten, classes=[0,1,2]) # [0,1,1], [0,1,0], len (2157568,) 
    print(y_test_flatten.shape, "yteste")
    y_score = torch.gather(y_score, 1, y_test) #[14, 1, 512, 301]
    
    y_score_flatten = y_score.flatten() #[2157568]
    
    n_classes =  y_test_flatten.shape[1] 
    print(n_classes, "nclases")
    

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_flatten[:, i], y_score_flatten[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_flatten.ravel(), y_score_flatten.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()

    roc_score = roc_auc["macro"]
    #wandb.log({"ROC": plt})
    #return roc_score
