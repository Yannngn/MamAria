import os
import torch
import wandb

from munch import munchify
from tqdm import tqdm
from yaml import safe_load

from utils.utils import save_checkpoint
from validate import validate_fn, early_stop_validation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))
PATH = os.path. dirname(__file__)

def train_fn(loader, model, optimizer, loss_fn, scaler, config, device=DEVICE):
    print(f'='.center(125, '='))
    print('Training model... \n')
    
    loop = tqdm(loader, bar_format='{l_bar}{bar:75}{r_bar}{bar:-75b}')
    closs = 0

    for _, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.long().to(device)
        #print(targets.unique(), "targets")
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        # wandb logging
        wandb.log({"batch loss":loss.item()})
        closs += loss.item()
    
    wandb.log({"loss":closs/config.BATCH_SIZE})
    loop.close()
    return loss.item()

def train_loop(train_loader, val_loader, model, optimizer, scheduler, loss_fn, scaler, stopping, global_metrics, label_metrics, config, load_epoch=0, time=0, device=DEVICE):  
    for epoch in range(load_epoch, CONFIG.HYPERPARAMETERS.MAX_NUM_EPOCHS):      
        print(f'='.center(125, '='))
        print(f'   BEGINNING EPOCH {epoch}:   '.center(125,'='))       
        
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, config, device)

        # save model
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
                
        save_checkpoint(checkpoint)

        # check accuracy
        
        print(f'='.center(125, '='))
        print("Validating results ... \n")
        
        val_loss = validate_fn(val_loader, model, loss_fn, scheduler, train_loss, epoch, time, global_metrics, label_metrics)
        
        if not CONFIG.PROJECT.EARLYSTOP: continue
        
        stopping(val_loss, checkpoint, checkpoint_path=PATH+f"/data/checkpoints/{time}_best_checkpoint.pth.tar", epoch=epoch)
        
        if stopping.early_stop: 
            early_stop_validation(val_loader, model, train_loss, val_loss, epoch, time, device)
            wandb.run.finish()
            break
        
    early_stop_validation(val_loader, model, train_loss, val_loss, global_metrics, label_metrics, epoch, time)
        
    
    print(f'='.center(125, '='))
    print("Training Finished")

    wandb.run.finish()