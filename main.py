import logging
import os
import torch
import wandb
import warnings

from datetime import datetime
from munch import munchify, unmunchify
from yaml import safe_load

from models.unet import UNET
from train import train_loop
from utils.early_stopping import EarlyStopping
from utils.utils import (get_device, get_metrics, get_scheduler, 
                         load_checkpoint, get_loaders, get_loss_function, 
                         get_optimizer, get_transforms)

def main(config):
    device = get_device(config)

    logging.info(torch.cuda.is_available())
    logging.info(torch.cuda.device_count())
    logging.info(torch.cuda.current_device())

    wandb.init(
        project = config.wandb.project_name,
        entity = config.wandb.project_team,
        config = unmunchify(config.hyperparameters)
    )

    ### utilizar parametros do sweep caso tenha
    config.hyperparameters = munchify(wandb.config)

    #train_transforms, val_transforms, test_transforms = get_transforms(config)
    # train_loader, val_loader, _ = get_loaders(config, train_transforms, val_transforms, test_transforms)

    train_loader, val_loader, _ = get_loaders(config, *get_transforms(config)) ### Testar isso

    model = UNET(config).to(device)
    model = torch.nn.DataParallel(model)

    loss_fn = get_loss_function(config)
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_scheduler(config, optimizer)
    scaler = torch.cuda.amp.GradScaler()   
    stopping = EarlyStopping(patience=config.hyperparameters.earlystop_patience, 
                             wait=config.hyperparameters.earlystop_wait
                             ) if config.hyperparameters.earlystopping else None 
    
    config.project.epoch = load_checkpoint(torch.load(config.load.path), 
                                           model, 
                                           optimizer, 
                                           scheduler
                                           ) if config.project.load_model else 0

    global_metrics, label_metrics = get_metrics(config)
    
    logging.info('entering train')
    train_loop(train_loader, val_loader, model, 
               optimizer, scheduler, loss_fn,
               scaler, stopping,
               global_metrics, label_metrics, config)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings("ignore")
    
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    
    with open('config.yaml') as f:
        config = munchify(safe_load(f))  

    os.environ['WANDB_MODE'] = 'online' if config.wandb.online else 'offline'
    config.project.time = datetime.now().strftime("%Y%m%d_%H%M%S")

    main(config)

