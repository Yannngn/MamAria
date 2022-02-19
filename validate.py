import torch

from munch import munchify
from tqdm import tqdm
from yaml import safe_load

from utils.logs import log_predictions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))

def validate_fn(val_loader, model, loss_fn, scheduler, train_loss, epoch, time):
    loop = tqdm(val_loader, bar_format='{l_bar}{bar:75}{r_bar}{bar:-75b}')
    model.eval()
    
    for _, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.long().to(DEVICE)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        if CONFIG.PROJECT.SCHEDULER:
            scheduler.step(loss.item())
   
    if epoch % CONFIG.PROJECT.VALIDATION_INTERVAL == 0:
        print(f'='.center(125, '='))
        print("   Logging and saving predictions...   ".center(125, '='))
        log_predictions(val_loader, model, train_loss, loss.item(), epoch, time=time)

    model.train()
    return loss.item()