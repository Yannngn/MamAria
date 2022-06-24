import os
import torch
import torch.nn as nn
import wandb
import warnings

from datetime import datetime
from munch import munchify, unmunchify
from tqdm import tqdm
from yaml import safe_load

from loggers.logs import log_predictions
from models.unet import UNET
from utils.utils import get_loaders, load_checkpoint, get_device, get_metrics, get_transforms, get_loss_function

warnings.filterwarnings("ignore")

def predict_fn(test_loader, model, loss_fn, global_metrics, label_metrics, config):
    device = get_device(config)
    loop = tqdm(test_loader)

    model.eval()

    for idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.long().to(device)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
        log_predictions(data, targets, predictions, global_metrics, label_metrics, config, idx)
    
    model.train()

    return loss.item()

def main(config):
    device = get_device(config)
   # logging.info('predict')
    wandb.init(
        project = config.wandb.project_name,
        entity = config.wandb.project_team,
        config = unmunchify(config.hyperparameters))
    
    #logging.info('wandb init')
    config.hyperparameters = munchify(wandb.config)

    train_transforms, val_transforms, test_transforms = get_transforms(config)
    global_metrics, label_metrics = get_metrics(config)
    _, _, test_loader = get_loaders(config, train_transforms,val_transforms, test_transforms)

    model = UNET(config).to(device)
    model = nn.DataParallel(model)
    
    loss_fn = get_loss_function(config)

    load_checkpoint(torch.load("data/checkpoints/20220619_061736_best_checkpoint.pth.tar"), model, optimizer=None, scheduler=None)

#load_checkpoint(checkpoint, model, optimizer, scheduler)
    predict_fn(
        
        test_loader,
        model,
        loss_fn,
        global_metrics, 
        label_metrics,
        config,
    )
    
    wandb.finish()  

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    with open('config.yaml') as f:
        config = munchify(safe_load(f))  

    os.environ['WANDB_MODE'] = 'online' if config.wandb.online else 'offline'
    config.time = datetime.now().strftime("%Y%m%d_%H%M%S")

    main(config)