import matplotlib as mpl
import random
import torch

from torch import nn

from datetime import datetime
from matplotlib import pyplot as plt
from munch import munchify
from scipy.special import softmax
from tqdm import tqdm
from yaml import safe_load

from pycalib.visualisations import plot_reliability_diagram
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss
from typing import Tuple, List, Optional
                                     
from models.unet import UNET
from utils import metrics
from utils.utils import get_device, get_loaders, get_loss_function, get_transforms, load_checkpoint

mpl.use('Agg')

def main(config):
    device = get_device(config)
    
    _, val_transforms, calib_transforms = get_transforms(config) 
    _, val_loader, calib_loader = get_loaders(config, None, val_transforms, calib_transforms)

    model = UNET(config).to(device)
    model = nn.DataParallel(model)
    load_checkpoint(torch.load(config.load.path, map_location=torch.device('cpu')), model, None, None)
    
    loss_fn = get_loss_function(config)

    logits, labels = predict(model, val_loader, loss_fn, device)

    print_metrics(logits, labels)
   
    odir = False
    lambda_ = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    mu_ = lambda_ if odir else None

    sample = random.sample(range(0, torch.numel(labels)), 1 * torch.numel(labels) // (1 * 6))
    gscv = fit_calibrator(logits, labels, lambda_, mu_, sample)
    #gscv = fit_calibrator(logits, labels, lambda_, mu_)
    
    logits, labels = predict(model, calib_loader, loss_fn, device)

    plot_results(gscv, logits, labels, odir)

def print_metrics(logits: torch.Tensor, labels:torch.Tensor) -> None:
    ece_criterion = metrics.ECELoss()
    #Torch version
    logits_np = logits.cpu().float().numpy()
    labels_np = labels.cpu().float().numpy()

    #Numpy Version
    print(f'ECE: {ece_criterion.loss(logits_np,labels_np,15):.3f}')

    mce_criterion = metrics.MCELoss()
    print(f'MCE: {mce_criterion.loss(logits_np,labels_np):.3f}')

def fit_calibrator(logits: torch.Tensor, labels: torch.Tensor, lambda_: list, mu_: Optional[list], sample: Optional[list]=None):
    scores = logits.permute(0, 2, 3, 1)
    scores = scores.flatten(end_dim=2)
    scores = softmax(scores.cpu().float().numpy(), axis=1)
    labels = labels.flatten().cpu().float().numpy()

    calibrator = FullDirichletCalibrator(reg_lambda=lambda_, reg_mu=mu_)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    gscv = GridSearchCV(calibrator, 
                        cv=skf,
                        scoring='neg_log_loss',
                        param_grid={'reg_lambda': lambda_,
                                    'reg_mu': mu_ if mu_ else [None]
                                    },
                        verbose=1
                        )

    gscv.fit(scores[sample], labels[sample])

    print('Grid of parameters cross-validated')
    print(gscv.param_grid)
    print(f'Best parameters: {gscv.best_params_}')

    return gscv

def predict(model, loader, loss_fn, device) -> Tuple[torch.Tensor, torch.Tensor]:
    loop = tqdm(loader)
    logits_list, labels_list = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.long().to(device)
            #outputs are the the raw scores!
            logits = model(images)
            #add data to list
            logits_list.append(logits)
            labels_list.append(labels)
            loss = loss_fn(logits, labels)
            #convert to probabilities
            output_probs = nn.functional.softmax(logits, dim=1)
            #get predictions from class
            _, predicted = torch.max(output_probs, 1)
            #total
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())

    loop.close()

    pixels = config.image.image_height * config.image.image_width
    print(f'Accuracy of the network on the {total * pixels} test pixels: {100 * correct / (total * pixels):.3f} %')

    return torch.cat(logits_list), torch.cat(labels_list)

def plot_results(model, logits: torch.Tensor, labels: torch.Tensor, odir: bool) -> None:
    logits = logits.permute(0, 2, 3, 1)
    logits = logits.flatten(end_dim=2)
    scores = softmax(logits.detach().cpu().numpy(), axis=1)
    
    l = labels.flatten().detach().cpu().numpy()

    fig = plot_reliability_diagram(labels=l, scores=scores,
                                   class_names=['Background', 'Low', 'Mid', 'High'],
                                   show_gaps=True, 
                                   show_bars=True,
                                   show_histogram=True
                                   )
    
    plt.title(f'Before {"Odir" if odir else "Full"} Reliability Diagram')
    plt.savefig(f'plots/{NOW}_pre_{"odir" if odir else "full"}.png')
    
    loss = log_loss(l, scores)
    print(f"TEST log-loss: UNET {loss:.2f}")

    results = model.predict_proba(scores)

    loss = log_loss(l, results)
    print(f"TEST log-loss: Calibrator {loss:.2f}")

    fig = plot_reliability_diagram(labels=l, scores=results,
                                   class_names=['Background', 'Low', 'Mid', 'High'],
                                   show_gaps=True, 
                                   show_bars=True,
                                   show_histogram=True)

    plt.title(f'After {"Odir" if odir else "Full"} Reliability Diagram')
    plt.savefig(f'plots/{NOW}_pos_{"odir" if odir else "full"}.png')    

if __name__ == '__main__':
    NOW = datetime.now().strftime("%m%d%Y-%H%M%S")
    with open('config_prediction.yaml') as f:
        config = munchify(safe_load(f))  

    main(config)
    


    



