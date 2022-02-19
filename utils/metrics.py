import numpy as np
import sklearn.metrics
import torch

from utils.post_processing import fit_ellipses_on_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_iou(pred: np.array, label: np.array) -> np.float32:
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

def dice_coef(y_pred: torch.tensor, y_true: torch.tensor) -> np.float32:

    intersection = np.sum(y_true * y_pred)
    smooth = 0.0001
    
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def dice_coef_multilabel(y_pred: torch.tensor, y_true: torch.tensor) -> list[np.float32]:
    num_labels = len(np.unique(y_true))
    dice=[]
    
    for index in range(num_labels):
        dice.append(dice_coef(y_true == index, y_pred == index))

    return dice

def roc_auc_multilabel(y_pred: torch.tensor, y_true: torch.tensor) -> tuple[list, list]:
    num_labels = len(np.unique(y_true))
    auc_, curve_= [],[]

    print(y_pred.shape)
    for index in range(num_labels):
        temp_pred = y_pred[:,index,:,:].flatten()
        temp_true = np.zeros_like(y_true)
        temp_true[y_true == index] = 1

        curve_.append(list(sklearn.metrics.precision_recall_curve(temp_true, temp_pred)))
        #auc_.append(roc_auc_score(temp_true, temp_pred))
        auc_.append(sklearn.metrics.average_precision_score(temp_true, temp_pred))
        
    return auc_, curve_

def macro_roc_curve(lst: list) -> tuple[np.array, np.array, float]:
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

    return all_fpr, mean_tpr, sklearn.metrics.auc(all_fpr, mean_tpr)

def roc_auc_multilabel_global(y_pred: torch.tensor, y_true: torch.tensor):
    num_labels = len(np.unique(y_true))

    flatten_y_pred = []
    flatten_y_true = []
    for index in range(num_labels):
        flatten_y_pred.append(y_pred[:,index,:,:].flatten())
        temp = np.zeros_like(y_true)
        temp[y_true == index] = 1
        flatten_y_true.append(temp)
    flatten_y_pred=np.array(flatten_y_pred)
    weighted_roc_auc_ovo = sklearn.metrics.roc_auc_score(
        flatten_y_true, flatten_y_pred, multi_class="ovo", average="weighted"
    )
    
    weighted_roc_auc_ovr = sklearn.metrics.roc_auc_score(
        flatten_y_true, flatten_y_pred, multi_class="ovr", average="weighted"
    )
    
    return weighted_roc_auc_ovo, weighted_roc_auc_ovr

def compute_global_accuracy(pred: np.array, label: np.array) -> float:
    count = 0.

    for l, lab in enumerate(label):
        if pred[l] == lab: 
            continue
        
        count += 1.

    return count / len(label)

def evaluate_segmentation(prob: torch.tensor, label: torch.tensor, score_averaging=None) -> dict:

    pred = torch.argmax(prob, 1)
    
    pred_ = pred.unsqueeze(1).cpu().numpy()
    label_ = label.unsqueeze(1).cpu().numpy()

    flat_pred = pred_.flatten()
    flat_label = label_.flatten()
    
    prob = prob.cpu().numpy()
    
    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    dict_eval = {"accuracy":global_accuracy}
 
    acc = sklearn.metrics.jaccard_score(flat_pred, flat_label, average=score_averaging)
    prec = sklearn.metrics.precision_score(flat_pred, flat_label, average=score_averaging)
    rec = sklearn.metrics.recall_score(flat_pred, flat_label, average=score_averaging, zero_division = 0)

    pred_eli = fit_ellipses_on_image(pred)
    flat_pred_eli = pred_eli.flatten()
    label_eli = fit_ellipses_on_image(label)
    flat_label_eli = label_eli.flatten()

    ellipse_global_accuracy = compute_global_accuracy(flat_pred_eli, flat_label_eli)
    ellipse_acc = sklearn.metrics.jaccard_score(flat_pred_eli, flat_label_eli, average=score_averaging)
    ellipse_prec = sklearn.metrics.precision_score(flat_pred_eli, flat_label_eli, average=score_averaging)
    ellipse_rec = sklearn.metrics.recall_score(flat_pred_eli, flat_label_eli, average=score_averaging, zero_division = 0)

    dict_eval["accuracy_ellipse"] = ellipse_global_accuracy

    i = 2
    dict_eval[f'accuracy_label_{i}'] = acc[i]
    dict_eval[f'precision_label_{i}'] = prec[i]
    dict_eval[f'recall_label_{i}'] = rec[i]
    dict_eval[f'accuracy_ellipse_label_{i}'] = ellipse_acc[i]
    dict_eval[f'precision_ellipse_label_{i}'] = ellipse_prec[i]
    dict_eval[f'recall_ellipse_label_{i}'] = ellipse_rec[i]

    return dict_eval
