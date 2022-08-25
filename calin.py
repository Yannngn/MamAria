from cProfile import label
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.patches as mpatches
import torch.nn as nn
import matplotlib as mpl
from torch import optim
from matplotlib import pyplot as plt
from munch import munchify
from scipy.special import softmax
from tqdm import tqdm
from yaml import safe_load
from models.unet import UNET
from utils import metrics
from utils import temperature as recalibration
from utils.utils import get_device, get_loaders, get_loss_function, get_transforms, load_checkpoint
from visualization import calibration as visualization



################### Testing ######################

# Use kwags for calibration method specific parameters
def test(x, calibration_method=None, **kwargs):
    device = get_device(x)
    
    _, _, transforms = get_transforms(x) 
    _, val_loader, loader = get_loaders(x, None, None, transforms)

    model = UNET(x).to(device)
    model = nn.DataParallel(model)
    load_checkpoint(torch.load(x.load.path, map_location=torch.device('cpu')), model, None, None)
    
    loss_fn = get_loss_function(x)

    correct, total = 0, 0

    # First: collect all the logits/preds and labels for the validation set
    preds, labels_oneh, losses, logits_list, labels_list = [], [], [], [], []
    
    loop = tqdm(loader)

    with torch.no_grad():
        for images, labels in tqdm(loop):
            images, labels = images.to(device), labels.long().to(device)
            pred = model(images)
            loss = loss_fn(pred, labels)
            if calibration_method:
                pred = calibration_method(pred, kwargs)
                loss = loss_fn(T_scaling(pred, kwargs), labels)                
                optimizer = optim.LBFGS([kwargs], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')


            # Get softmax values for net input and resulting class predictions
            pred = nn.functional.softmax(pred, dim=1)
            #get predictions from class
            _, predicted_cl = torch.max(pred, 1)
            pred = pred.cpu().detach().numpy()

            # Convert labels to one hot encoding
            label_oneh = torch.nn.functional.one_hot(labels, num_classes=4)
            label_oneh = label_oneh.cpu().detach().numpy()

            preds.extend(pred)
            labels_oneh.extend(label_oneh)
            losses.append(loss)

            # Count correctly classified samples for accuracy
            correct += sum(predicted_cl == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())
            
        loop.close()

    for images, labels in enumerate(tqdm(val_loader, 0)):
        images, labels = images.to(device), labels.long().to(device)

        model.eval()
        with torch.no_grad():
            logits_list.append(model(images))
            labels_list.append(labels)

    preds = np.array(preds).flatten()
    labels_oneh = np.array(labels_oneh).flatten()
    
    logits_list = torch.cat(preds).to(device)
    labels_list = torch.cat(labels_oneh).to(device)

    #correct_perc = correct / len(test_set)
    #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_perc))
    print(f'Accuracy of the network on the {total * x.image.image_height * x.image.image_width} test pixels: {100 * correct / (total * x.image.image_height * x.image.image_width):.3f} %')

    return preds, labels_oneh, losses, logits_list, labels_list

#preds, labels_oneh = test()

def T_scaling(logits, args):
    temperature = args.get('temperature', None)
    return torch.div(logits, temperature)

def calc_bins(preds):
    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE


def draw_reliability_graph(preds):
    ECE, MCE = get_metrics(preds)
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
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
    MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
    plt.legend(handles=[ECE_patch, MCE_patch])

    #plt.show()
    
    plt.savefig('calibrated_network.png', bbox_inches='tight')

    #draw_reliability_graph(preds)


def main (config):
    preds, labels_oneh, losses, logits_list, labels_list = test(config)
    # criterion = nn.CrossEntropyLoss()
    temperature = nn.Parameter(torch.ones(1).cuda())
    args = {'temperature': temperature}
    temps = []
    temps.append(temperature.item())

    print('Final T_scaling factor: {:.2f}'.format(temperature.item()))

    plt.subplot(121)
    plt.plot(list(range(len(temps))), temps)

    plt.subplot(122)
    plt.plot(list(range(len(losses))), losses)
    plt.show()

    preds_original, _ = test(x=config)
    preds_calibrated, _ = test(x=config, calibration_method=T_scaling, temperature=temperature)

    draw_reliability_graph(preds_original)
    draw_reliability_graph(preds_calibrated)

if __name__ == '__main__':
    with open('config_prediction.yaml') as f:
        config = munchify(safe_load(f))  

    main(config)