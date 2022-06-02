import torch
from sklearn import metrics

from utils.post_processing import fit_ellipses_on_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_segmentation(prob: torch.tensor, label: torch.tensor, score_averaging=None) -> dict:
    num_classes = int(torch.max(label).item() + 1)
    
    pred = torch.argmax(prob, 1)
    label = label.long()

    flat_pred, flat_label = pred.flatten(), label.flatten()
    flat_pred_cpu, flat_label_cpu = flat_pred.cpu().numpy(), flat_label.cpu().numpy()    
    
    dict_eval = {}
    
    #jac = tmf.jaccard_index(prob, label, num_classes=num_classes, reduction=score_averaging)
    #acc = tmf.accuracy(prob, label, average=score_averaging, mdmc_average=score_averaging, num_classes=num_classes)
    #prec, rec = tmf.precision_recall(prob, label, average=score_averaging, mdmc_average=score_averaging, num_classes=num_classes)
    
    acc = metrics.accuracy_score(flat_label_cpu, flat_pred_cpu, average=score_averaging)
    jac = metrics.jaccard_score(flat_label_cpu, flat_pred_cpu, average=score_averaging)
    prec = metrics.precision_score(flat_label_cpu, flat_pred_cpu, average=score_averaging, zero_division = 0)
    rec = metrics.recall_score(flat_label_cpu, flat_pred_cpu, average=score_averaging, zero_division = 0)
    '''
    pred_eli = fit_ellipses_on_image(pred)
    flat_pred_eli = pred_eli.flatten()
    label_eli = fit_ellipses_on_image(label)
    flat_label_eli = label_eli.flatten()

    ellipse_global_accuracy = compute_global_accuracy(flat_pred_eli, flat_label_eli)
    ellipse_acc = sklearn.metrics.jaccard_score(flat_pred_eli, flat_label_eli, average=score_averaging)
    ellipse_prec = sklearn.metrics.precision_score(flat_pred_eli, flat_label_eli, average=score_averaging)
    ellipse_rec = sklearn.metrics.recall_score(flat_pred_eli, flat_label_eli, average=score_averaging, zero_division = 0)

    dict_eval["accuracy_ellipse"] = ellipse_global_accuracy
    '''
    for i in range(num_classes):
        dict_eval[f'accuracy_label_{i}'] = acc[i]
        dict_eval[f'accuracy_label_{i}'] = jac[i]
        dict_eval[f'precision_label_{i}'] = prec[i]
        dict_eval[f'recall_label_{i}'] = rec[i]
        
    return dict_eval
