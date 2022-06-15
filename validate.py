import torch

from munch import munchify
from tqdm import tqdm
from yaml import safe_load

from loggers.logs import log_predictions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))

def validate_fn(val_loader, model, loss_fn, scheduler, train_loss, epoch, time, global_metrics, label_metrics):
    loop = tqdm(val_loader, bar_format='{l_bar}{bar:75}{r_bar}{bar:-75b}')
    model.eval()
    
    for idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.long().to(DEVICE)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        scheduler.step(loss.item())
        
        log_predictions(predictions, targets, train_loss, loss.item(), global_metrics, label_metrics, idx, epoch, time=time)

    model.train()
    return loss.item()