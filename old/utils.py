import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from sklearn.metrics import precision_score, recall_score, roc_auc_score, jaccard_score, auc, precision_recall_curve, average_precision_score
import os
import cv2
from tqdm import tqdm
import numpy as np
from yaml import safe_load
from munch import munchify
import wandb
from datetime import datetime
import json

from utils.dataset import PhantomDataset

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
        mask = cv2.imread(mask)
    
        if total_pixels == 0:
            total_pixels = mask.shape[1] * mask.shape[2]

        temp = []
        
        for i in range(num_labels):
            temp.append((mask == i).sum())
        
        weights += temp
    den = weights / (total_pixels * len(mask))
    out = np.divide(multiplier, den, out = np.zeros_like(multiplier, dtype = float), where = den!=0)
    return torch.tensor(out).float().to(device)

def check_accuracy(loader, model, device=DEVICE):
    loop = tqdm(loader)
    num_correct = 0
    num_pixels = 0
    
    model.eval()
    with torch.no_grad():
        for x, y in loop:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            
            y_ = y.unsqueeze(1)

            preds_labels = torch.argmax(preds, 1).unsqueeze(1) #(14, 1, 512, 301) -> flatten (2157568,)
                
            num_correct += (preds_labels == y_).sum()
            num_pixels += torch.numel(preds_labels)

            dict_eval = evaluate_segmentation(preds, y, score_averaging = None)
    model.train()

    return num_correct, num_pixels, dict_eval

def log_predictions(val_loader, model, loss_train, loss_val, epoch, time=0, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=DEVICE):
    num_correct, num_pixels, dict_eval  = check_accuracy(val_loader, model, device)

    dict_eval['epoch'] = epoch
    dict_eval['loss_train'] = loss_train
    dict_eval['loss_val'] = loss_val

    print(f"Got {num_correct} of {num_pixels} pixels;")
    #for key in dict_eval:
    #    print (key,':', dict_eval[key])

    json_lines = json.dumps(dict_eval)
    
    with open(folder+f'{time}_prediction.json','w') as f:
        json.dump(json_lines, f)  
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_predictions_as_imgs(val_loader, model, epoch, dict_eval, time=now, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=device)
    save_ellipse_pred_as_imgs(val_loader, model, epoch, dict_eval, time=now, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=device)

def log_submission(loader, model, loss_test, time=0, folder=CONFIG.PATHS.SUBMISSIONS_DIR, device = DEVICE):
    num_correct, num_pixels, dict_subm = check_accuracy(loader, model, device)
    dict_subm['loss_subm'] = loss_test

    print(f"Got {num_correct} of {num_pixels} pixels;")
    # for key in dict_subm:
    #     print (key,':', dict_subm[key])
    
    #json_lines = json.dumps(dict_subm)
    
    with open(folder+f'{time}_submission.json','w') as f:
        json.dump(dict_subm, f, ensure_ascii=False, indent=4)  

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_submission_as_imgs(loader, model, dict_subm, time=now, folder=CONFIG.PATHS.SUBMISSIONS_DIR, device=device)

def save_predictions_as_imgs(loader, model, epoch, dict_eval, folder=CONFIG.PATHS.PREDICTIONS_DIR, time=0, device=DEVICE):
    print("=> Saving predictions as images ...")
    
    model.eval()
    with torch.no_grad():    
        for idx, (x, _) in enumerate(loader):
            x = x.to(device)
            preds_labels = torch.argmax(model(x), 1)
            preds_labels = label_to_pixel(preds_labels)
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

            for j in range(preds_labels.size(0)):
                img = folder + f"{time}_submission_i{idx:02d}_p{j:02d}.png"
                save_image(preds_labels[j, :, :, :], img)
                dict_subm[f'submission_i{idx:02d}_p{j:02d}'] = wandb.Image(img)
            
    wandb.log(dict_subm)

def save_validation_as_imgs(loader, folder=CONFIG.PATHS.PREDICTIONS_DIR, time=0, device=DEVICE):
    print("=> Saving validation images ...")
    dict_val = {}
    for idx, (_, y) in enumerate(loader):
        y = y.to(device)
        val = (y / y.max()).unsqueeze(1)
        img = f"{folder}{time}_val_i{idx:02d}.png"
        save_image(val, img)
        dict_val[f'validation_i{idx:02d}'] = wandb.Image(img)
    
    wandb.log(dict_val)

def save_test_as_imgs(loader, folder=CONFIG.PATHS.PREDICTIONS_DIR, time=0, device=DEVICE):
    print("=> Saving test images ...")
    dict_val = {}
    for idx, (_, y) in enumerate(loader):
        y = y.to(device)
        val = (y / y.max()).unsqueeze(1)
        for j in range(y.size(0)):
            img = f"{folder}{time}_test_i{idx:02d}_p{j:02d}.png"
            save_image(val, img)
            dict_val[f'test_i{idx:02d}_p{j:02d}'] = wandb.Image(img)
    
    wandb.log(dict_val)

def label_to_pixel(preds, col='l'):
    if col == 'l':
        preds = preds / (CONFIG.IMAGE.MASK_LABELS - 1) #0, 1, 2 
        preds = preds.unsqueeze(1).float()
        return preds

    else:
        preds = preds[:,1:,:,:]
        return preds.float()

def compute_iou(pred, label):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        
        pred_i = pred == val
        label_i = label == val
        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    return I / U

def dice_coef(y_pred,y_true):

    intersection = np.sum(y_true * y_pred)
    smooth = 0.0001
    
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def dice_coef_multilabel(y_pred,y_true):
    num_labels = len(np.unique(y_true))
    dice=[]
    
    for index in range(num_labels):
        dice.append(dice_coef(y_true == index, y_pred == index))

    return dice

def roc_auc_multilabel(y_pred:torch.tensor,y_true:torch.tensor)->tuple[list, list]:
    num_labels = len(np.unique(y_true))
    auc_, curve_= [],[]

    print(y_pred.shape)
    for index in range(num_labels):
        temp_pred = y_pred[:,index,:,:].flatten()
        temp_true = np.zeros_like(y_true)
        temp_true[y_true == index] = 1

        curve_.append(list(precision_recall_curve(temp_true, temp_pred)))
        #auc_.append(roc_auc_score(temp_true, temp_pred))
        auc_.append(average_precision_score(temp_true, temp_pred))
        
    return auc_, curve_

def macro_roc_curve(lst:list)->tuple[np.array, np.array, float]:
    n_classes = len(lst)

    fpr =[l[0] for l in lst]
    tpr = [l[1] for l in lst]
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

def roc_auc_multilabel_global(y_pred,y_true):
    num_labels = len(np.unique(y_true))

    flatten_y_pred = []
    flatten_y_true = []
    for index in range(num_labels):
        flatten_y_pred.append(y_pred[:,index,:,:].flatten())
        temp = np.zeros_like(y_true)
        temp[y_true == index] = 1
        flatten_y_true.append(temp)
    flatten_y_pred=np.array(flatten_y_pred)
    weighted_roc_auc_ovo = roc_auc_score(
        flatten_y_true, flatten_y_pred, multi_class="ovo", average="weighted"
    )
    
    weighted_roc_auc_ovr = roc_auc_score(
        flatten_y_true, flatten_y_pred, multi_class="ovr", average="weighted"
    )
    
    return weighted_roc_auc_ovo, weighted_roc_auc_ovr

def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

def evaluate_segmentation(prob, label, score_averaging=None):

    pred = torch.argmax(prob, 1)
    pred_ = pred.unsqueeze(1).cpu().numpy()
    
    label_ = label.unsqueeze(1).cpu().numpy()

    flat_pred = pred_.flatten()#.cpu().numpy()
    flat_label = label_.flatten()#.cpu().numpy()
    
    prob = prob.cpu().numpy()
    #pred = pred.cpu().numpy()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    acc = jaccard_score(flat_pred, flat_label, average=score_averaging)
    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging, zero_division = 0)
    #iou = compute_iou(flat_pred, flat_label)
    #dice = dice_coef_multilabel(flat_pred, flat_label)
    pred_eli = fit_ellipses_on_image(pred)
    flat_pred_eli = pred_eli.flatten()#.cpu().numpy()
    label_eli = fit_ellipses_on_image(label)
    flat_label_eli = label_eli.flatten()

    ellipse_global_accuracy = compute_global_accuracy(flat_pred_eli, flat_label_eli)
    ellipse_acc = jaccard_score(flat_pred_eli, flat_label_eli, average=score_averaging)
    ellipse_prec = precision_score(flat_pred_eli, flat_label_eli, average=score_averaging)
    ellipse_rec = recall_score(flat_pred_eli, flat_label_eli, average=score_averaging, zero_division = 0)
    #ellipse_iou = compute_iou(flat_pred, flat_label)
    #ellipse_dice = dice_coef_multilabel(flat_pred, flat_label)

    #rocauc, curve = roc_auc_multilabel(prob, flat_label)
    #macro_fpr, macro_tpr, macro_auc = macro_roc_curve(curve)

    dict_eval = {"accuracy":global_accuracy,
                 "ellipse_accuracy":ellipse_global_accuracy,
                 #"macro_fpr":macro_fpr.tolist(), 
                 #"macro_tpr": macro_tpr.tolist(),
                 #"macro_auc":macro_auc
                 }

    #for i in range(CONFIG.IMAGE.MASK_LABELS):
    i = 2
    dict_eval[f'accuracy_label_{i}'] = acc[i]
    dict_eval[f'precision_label_{i}'] = prec[i]
    dict_eval[f'recall_label_{i}'] = rec[i]
    #dict_eval[f'iou_label_{i}'] = iou[i]
    #dict_eval[f'auc_label_{i}'] = rocauc[i]
    #dict_eval[f'fpr_label_{i}'] = curve[i][0].tolist()
    #dict_eval[f'tpr_label_{i}'] = curve[i][1].tolist()
    #dict_eval[f'thresholds_label_{i}'] = curve[i][2].tolist()
    #dict_eval[f'ellipse_dice_label_{i}'] = ellipse_dice[i]
    dict_eval[f'ellipse_accuracy_label_{i}'] = ellipse_acc[i]
    dict_eval[f'ellipse_precision_label_{i}'] = ellipse_prec[i]
    dict_eval[f'ellipse_recall_label_{i}'] = ellipse_rec[i]

    return dict_eval

def fit_ellipses_on_image(image: np.array) -> np.array:
    lesion = (CONFIG.IMAGE.MASK_LABELS - 1)
    # recebe torch.argmax(pred, 1)
    image = image.cpu().numpy()
    image = image.astype(np.uint8)
    print(image.shape, np.max(image), np.median(image))
    # transforma em imagem
    original = image.copy()

    for i, img in enumerate(original):
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        _, thresh = cv2.threshold(src=blur, thresh=lesion-1, maxval=lesion, type=cv2.THRESH_BINARY)
        print(thresh.shape, np.max(thresh), np.median(thresh))

        cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            '''
            if len(c) >= 5:
                ellipse = cv2.fitEllipse(c)
                cv2.ellipse(img, ellipse, lesion, -1)
            '''
            convex = cv2.convexHull(c)
            cv2.drawContours(img, convex, -1, lesion, -1)

        image[i] = img

    return np.asarray(image, dtype=np.uint8)  

def save_ellipse_pred_as_imgs(loader, model, epoch, dict_eval, folder=CONFIG.PATHS.PREDICTIONS_DIR, time=0, device=DEVICE):
    print("=> Saving ellipses as images ...")
    
    model.eval()
    with torch.no_grad():    
        for idx, (x, _) in enumerate(loader):
            x = x.to(device)
            preds_labels = torch.argmax(model(x), 1)
            preds_labels = torch.tensor(fit_ellipses_on_image(preds_labels)).float().to(DEVICE)
            preds_labels = label_to_pixel(preds_labels)
            
            img = folder + f"{time}_elli_e{epoch}_i{idx}.png"
            save_image(preds_labels, img)
            dict_eval[f'elli_prediction_i{idx}'] = wandb.Image(img)
            
    model.train()
    wandb.log(dict_eval)

def save_ellipse_validation_as_imgs(loader, folder=CONFIG.PATHS.PREDICTIONS_DIR, time=0, device=DEVICE):
    print("=> Saving validation images ...")
    dict_val = {}
    for idx, (_, y) in enumerate(loader):
        print(y.shape, np.max(y), np.median(y))
        y = torch.tensor(fit_ellipses_on_image(y)).to(DEVICE)
        val = label_to_pixel(y)
        img = f"{folder}{time}_elli_val_i{idx:02d}.png"
        save_image(val, img)
        dict_val[f'elli_validation_i{idx:02d}'] = wandb.Image(img)
    
    wandb.log(dict_val)