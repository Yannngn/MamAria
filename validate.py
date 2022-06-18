import torch
import wandb

from munch import munchify
from tqdm import tqdm
from yaml import safe_load

from loggers.logs import log_predictions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))

def validate_fn(val_loader, model, loss_fn, scheduler, train_loss, epoch, time, global_metrics, label_metrics):
    print(f'='.center(125, '='))
    print("   Logging and saving predictions...   ".center(125, '='))    
    
    loop = tqdm(val_loader, bar_format='{l_bar}{bar:75}{r_bar}{bar:-75b}')
    model.eval()
    vloss = 0.
    for idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.long().to(DEVICE)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        scheduler.step(loss.item())
        
        wandb.log({"batch validation loss":loss.item()})
        vloss += loss.item()

        if ((epoch * len(val_loader) + idx) % CONFIG.PROJECT.VALIDATION_INTERVAL == 0):
            log_predictions(data, targets, predictions, global_metrics, label_metrics, idx, epoch, time=time)

    loop.close()
    wandb.log({"validation loss":vloss/CONFIG.HYPERPARAMETERS.BATCH_SIZE})
    model.train()
    return loss.item()

def early_stop_validation(val_loader, model, global_metrics, label_metrics, epoch, time=0):
    print(f'='.center(125, '='))
    print("Early Stopping ...")
    
    loop = tqdm(val_loader, bar_format='{l_bar}{bar:75}{r_bar}{bar:-75b}')
    model.eval()
    
    for idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.long().to(DEVICE)

        with torch.no_grad():
            predictions = model(data)
        
        log_predictions(data, targets, predictions, global_metrics, label_metrics, idx, epoch, time=time)
    
    loop.close()