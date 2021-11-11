import torch
import wandb
import os
from tqdm import tqdm
from munch import munchify
from yaml import safe_load

from utils import load_checkpoint, log_submission

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))
PATH = os.path.dirname(__file__)

def predict_fn(test_loader, model, loss_fn, time):
    loop = tqdm(test_loader)
    
    load_checkpoint(torch.load(PATH+f"/data/checkpoints/{time}_best_checkpoint.pth.tar"), model, optimizer=None, scheduler=None)

    model.eval()
    
    for idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.long().to(DEVICE)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
    log_submission(test_loader, model, loss.item(), time=time)
    
    model.train()
    
    wandb.finish()  

    return loss.item()