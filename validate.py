import torch
import wandb

from tqdm import tqdm

from loggers.logs import log_predictions
from utils.utils import get_device

def validate_fn(val_loader, model, loss_fn, scheduler, epoch, time, global_metrics, label_metrics, config):
    device = get_device(config)
    print(f'='.center(125, '='))
    print("   Logging and saving predictions...   ".center(125, '='))    
    
    loop = tqdm(val_loader, bar_format='{l_bar}{bar:75}{r_bar}{bar:-75b}')
    model.eval()
    vloss = 0.
    for idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.long().to(device)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        scheduler.step(loss.item())
        
        wandb.log({"batch validation loss":loss.item()})
        vloss += loss.item()

        if ((epoch * len(val_loader) + idx) % config.project.validation_interval == 0):
            log_predictions(data, targets, predictions, global_metrics, label_metrics, config, idx, epoch, time=time)

    loop.close()
    wandb.log({"validation loss":vloss/config.hyperparameters.batch_size})
    model.train()
    return loss.item()

def early_stop_validation(val_loader, model, global_metrics, label_metrics, config, epoch, time=0):
    device = get_device(config)
    print(f'='.center(125, '='))
    print("Early Stopping ...")
    
    loop = tqdm(val_loader, bar_format='{l_bar}{bar:75}{r_bar}{bar:-75b}')
    model.eval()
    
    for idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.long().to(device)

        with torch.no_grad():
            predictions = model(data)
        
        log_predictions(data, targets, predictions, global_metrics, label_metrics, config, idx, epoch, time=time)
    
    loop.close()