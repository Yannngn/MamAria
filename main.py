import logging
import os
import torch
import wandb
import warnings

from datetime import datetime

from munch import munchify, unmunchify
from torch.nn import DataParallel
from yaml import safe_load

from models.unet import UNET
from train import train_loop
from utils.early_stopping import EarlyStopping
from utils.utils import get_device, get_metrics, get_scheduler, load_checkpoint, get_loaders, get_loss_function, get_optimizer, get_transforms

def main(config):
    device = get_device(config)
    logging.info('main')
    wandb.init(
        project = config.wandb.project_name,
        entity = config.wandb.project_team,
        config = unmunchify(config.hyperparameters)
    )
    logging.info('wandb init')
    config.hyperparameters = munchify(wandb.config)

    train_transforms, val_transforms, test_transforms = get_transforms(config)
    logging.info('transforms')
    global_metrics, label_metrics = get_metrics(config)
    logging.info('metrics')
    # cross com 3 folds // 2 para treino 1 para calib


    ## média dos 3 modelos passando pelo calibs

    train_loader, val_loader, _ = get_loaders(config, train_transforms,val_transforms, test_transforms)
    logging.info('loader')
    model = UNET(config).to(device) #in_channels = CONFIG.IMAGE.IMAGE_CHANNELS, classes = CONFIG.IMAGE.MASK_LABELS, config = config).to(DEVICE)
    model = DataParallel(model)
    logging.info('model')
    loss_fn = get_loss_function(config)
    optimizer = get_optimizer(config, model.parameters())
       
    scheduler = get_scheduler(optimizer, config)
    
    config.epoch = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer, scheduler) if config.project.load_model else 0
        
    scaler = torch.cuda.amp.GradScaler()

    stopping = EarlyStopping(patience=config.hyperparameters.earlystopping_patience, wait=config.hyperparameters.earlystopping_wait)
    logging.info('optimizer and scheduler')

    #save_validation_as_imgs(val_loader, config)
    #save_ellipse_validation_as_imgs(val_loader, time = BEGIN)

    logging.info('entering train')
    train_loop(train_loader, val_loader, model, 
               optimizer, scheduler, loss_fn,
               scaler, stopping,
               global_metrics, label_metrics, config)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    
    with open('config.yaml') as f:
        config = munchify(safe_load(f))  

    os.environ['WANDB_MODE'] = 'online' if config.wandb.online else 'offline'
    config.time = datetime.now().strftime("%Y%m%d_%H%M%S")

    main(config)

