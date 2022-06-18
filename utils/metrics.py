import numpy as np
import torch

def check_accuracy(predictions, targets, global_metrics, label_metrics):
    dict_eval = evaluate_segmentation(predictions, targets, global_metrics, label_metrics)

    return dict_eval

def evaluate_segmentation(prob: torch.tensor, label: torch.tensor, global_metrics, label_metrics) -> dict:
    num_classes = prob.size(dim=1)
    dict_eval = {}

    for (metric, f) in global_metrics.items():
        #print(f"Calculating {metric} ...")
        result = f(prob, label).cpu().numpy()
        if result == np.nan: result = 0.
        dict_eval[f'global/{metric}'] = result
        
    for (metric, f) in label_metrics.items():
        result = f(prob, label).cpu().numpy()
        for i in range(num_classes):
            if result[i] == np.nan: result[i] = 0.
            dict_eval[f'label_{i}/{metric}'] = result[i]
        
    return dict_eval
