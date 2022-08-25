import matplotlib as mpl
import numpy as np
import torch

from torch import nn 
from torch import optim

from matplotlib import pyplot as plt
from matplotlib import patches
from munch import munchify
from scipy.special import softmax
from tqdm import tqdm
from yaml import safe_load

from calin import draw_reliability_graph
from models.unet import UNET
from utils import metrics
from utils.utils import get_device, get_loaders, get_loss_function, get_transforms, load_checkpoint
import pycalib
from pycalib.models import CalibratedModel, IsotonicCalibration

mpl.use('Agg')

def main(config):
    device = get_device(config)
    
    _, _, transforms = get_transforms(config) 
    _, _, loader = get_loaders(config, None, None, transforms)

    model = UNET(config).to(device)
    model = nn.DataParallel(model)
    load_checkpoint(torch.load(config.load.path, map_location=torch.device('cpu')), model, None, None)
    
    loss_fn = get_loss_function(config)

    #cal_clf = CalibratedModel(base_estimator=clf, method=cal) 



if __name__ == '__main__':
    with open('config_prediction.yaml') as f:
        config = munchify(safe_load(f))  

    main(config)
