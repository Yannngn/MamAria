import torch
from sklearn import metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_segmentation(prob: torch.tensor, label: torch.tensor, name='label', score_averaging=None) -> dict:
    num_classes = int(torch.max(label).item() + 1)
    
    pred = torch.argmax(prob, 1)
    label = label.long()

    flat_pred, flat_label = pred.flatten(), label.flatten()
    flat_pred_cpu, flat_label_cpu = flat_pred.cpu().numpy(), flat_label.cpu().numpy()    
    
    dict_eval = {}

    acc = metrics.accuracy_score(flat_label_cpu, flat_pred_cpu, average=score_averaging)
    jac = metrics.jaccard_score(flat_label_cpu, flat_pred_cpu, average=score_averaging)
    prec = metrics.precision_score(flat_label_cpu, flat_pred_cpu, average=score_averaging, zero_division = 0)
    rec = metrics.recall_score(flat_label_cpu, flat_pred_cpu, average=score_averaging, zero_division = 0)

    for i in range(num_classes):
        dict_eval[f'accuracy_{name}_{i}'] = acc[i]
        dict_eval[f'accuracy_{name}_{i}'] = jac[i]
        dict_eval[f'precision_{name}_{i}'] = prec[i]
        dict_eval[f'recall_{name}_{i}'] = rec[i]
        
    return dict_eval
