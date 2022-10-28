import logging
import torch
import wandb

from tqdm import tqdm

from loggers.logs import log_predictions
from utils.utils import get_device

def validate_fn(loader, model, loss_fn, scheduler, global_metrics, label_metrics, config):
    device = get_device(config)
    logging.info('Validating results...')    
    
    loop = tqdm(loader, bar_format='{l_bar}{bar:75}{r_bar}{bar:-75b}')
    vloss = 0.
    
    model.eval()
    for idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.long().to(device)

        # forward
        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        scheduler.step(loss.item())
        
        # wandb logging
        wandb.log({"batch validation loss":loss.item()})
        vloss += loss.item()

        # system logging
        if config.epoch != config.project.max_num_epochs - 1: continue
        if (config.epoch * len(loader) + idx) % config.project.validation_interval != 0: continue

        log_predictions(data, targets, predictions, global_metrics, label_metrics, config, idx)

    wandb.log({"validation loss":vloss/config.hyperparameters.batch_size})
    loop.close()
    
    return loss.item()

def early_stop_validation(loader, model, global_metrics, label_metrics, config):
    device = get_device(config)
    logging.info('Early stopping model...')
    
    loop = tqdm(loader, bar_format='{l_bar}{bar:75}{r_bar}{bar:-75b}')
    
    model.eval()
    for idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.long().to(device)

        with torch.no_grad():
            predictions = model(data)
        
        log_predictions(data, targets, predictions, global_metrics, label_metrics, config, idx)
    
    loop.close()