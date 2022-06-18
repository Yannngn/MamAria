import os
import torch
import wandb
import warnings

from datetime import datetime

from munch import munchify, unmunchify
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from yaml import safe_load

from models.unet import UNET
from train import train_loop
from utils.early_stopping import EarlyStopping
from utils.save_images import save_validation_as_imgs
from utils.utils import get_device, get_metrics, load_checkpoint, get_loaders, get_weights, get_loss_function, get_optimizer, get_transforms

def main(config, begin_time):
    device = get_device(config)

    wandb.init(
        project = config.project.project_name,
        entity = config.project.project_team,
        config = unmunchify(config.hyperparameters)
    )

    train_transforms, val_transforms, test_transforms = get_transforms(config)

    global_metrics, label_metrics = get_metrics(config)

    train_loader, val_loader, _ = get_loaders(config, train_transforms,val_transforms, test_transforms)

    model = UNET(config).to(device) #in_channels = CONFIG.IMAGE.IMAGE_CHANNELS, classes = CONFIG.IMAGE.MASK_LABELS, config = config).to(DEVICE)
    model = DataParallel(model)

    weights = get_weights(config)
    loss_fn = get_loss_function(config, weights)
    optimizer = get_optimizer(config, model.parameters())
       
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min') if config.project.scheduler else None 
    
    load_epoch = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer, scheduler) if config.project.load_model else 0
        
    scaler = torch.cuda.amp.GradScaler()

    stopping = EarlyStopping(patience=10, wait=30)
    
    save_validation_as_imgs(val_loader, config, time=begin_time)
    #save_ellipse_validation_as_imgs(val_loader, time = BEGIN)

    train_loop(
        train_loader, 
        val_loader, 
        model, 
        optimizer, 
        scheduler, 
        loss_fn, 
        scaler, 
        stopping,
        global_metrics, 
        label_metrics,
        config,
        load_epoch,
        time = begin_time
    )

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.cuda.empty_cache()
    os.environ['WANDB_MODE'] = 'offline'
    begin_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open('config.yaml') as f:
        config = munchify(safe_load(f))  
    
    main(config, begin_time)

