import numpy as np
import torch

from torch import nn 
from torch import optim
from torch.nn import DataParallel
from matplotlib import pyplot as plt
from matplotlib import patches
from munch import munchify
from tqdm import tqdm
from yaml import safe_load

from models.unet import UNET
from predict import main as predict
from utils.utils import get_device, get_loaders, get_metrics, get_transforms

#   preds = np.array(preds).flatten()
#   label_np = np.array(label_np).flatten()

def calc_bins(preds, label_np):
    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs, bin_confs, bin_sizes = np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (label_np[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds):
    ece = 0
    mce = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ece += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        mce = max(mce, abs_conf_dif)

    return ece, mce

def draw_reliability_graph(preds):
    ece, mce = get_metrics(preds)
    bins, _, bin_accs, _, _ = calc_bins(preds)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True) 
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
    plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # ECE and MCE legend
    ece_patch = patches.Patch(color='green', label=f'ECE = {100 * ece:.2f}%')
    mce_patch = patches.Patch(color='red', label=f'MCE = {100 * mce:.2f}%')
    plt.legend(handles=[ece_patch, mce_patch])
    
    plt.savefig('calibrated_network.png', bbox_inches='tight')

def T_scaling(logits, args):
    temperature = args.get('temperature', None)
    return torch.div(logits, temperature)

def main(config):
    def _eval():
        loss = criterion(T_scaling(logits_list, args), labels_list)
        loss.backward()
        temps.append(temperature.item())
        losses.append(loss)
        return loss
      
    device = get_device(config)

    _, val_transforms, _ = get_transforms(config)
    
    _, val_loader, _ = get_loaders(config, None, val_transforms, None)

    model = UNET(config).to(device)
    model = DataParallel(model)
    
    temperature = nn.Parameter(torch.ones(1).to(device))
    args = {'temperature': temperature}
    criterion = nn.CrossEntropyLoss()

    # Removing strong_wolfe line search results in jump after 50 epochs
    optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

    logits_list, labels_list, temps, losses = [], [], [], []

    for _, data in enumerate(tqdm(val_loader, 0)):
        images, labels = data[0].to(device), data[1].long().to(device)

        model.eval()
        with torch.no_grad():
            logits_list.append(model(images))
            labels_list.append(labels)

    # Create tensors
    logits_list = torch.cat(logits_list).to(device)
    labels_list = torch.cat(labels_list).to(device)

    optimizer.step(_eval)

    print(f'Final T_scaling factor: {temperature.item():.2f}')

    plt.subplot(121)
    plt.plot(list(range(len(temps))), temps)

    plt.subplot(122)
    plt.plot(list(range(len(losses))), losses.detach().numpy())
    plt.show()

    preds_original, _ = predict() #test def
    preds_calibrated, _ = predict(T_scaling, temperature=temperature)

    draw_reliability_graph(preds_original)
    draw_reliability_graph(preds_calibrated)

if __name__ == '__main__':
    with open('config.yaml') as f:
        config = munchify(safe_load(f))  
        
    main(config)

    



