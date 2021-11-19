import os
import torch, wandb
from tqdm import tqdm
from munch import munchify
from yaml import safe_load

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))
PATH = os.path. dirname(__file__)

from validate import validate_fn
from utils import log_predictions, save_checkpoint, check_accuracy

def train_fn(loader, model, optimizer, loss_fn, scaler, config):
    loop = tqdm(loader)
    closs = 0

    for _, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
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

    return loss.item()

def train_loop(train_loader, val_loader, model, optimizer, scheduler, loss_fn, scaler, stopping, config, load_epoch=0, time=0):
    #check_accuracy(train_loader, model, device=DEVICE)
    
    for epoch in range(load_epoch, CONFIG.HYPERPARAMETERS.NUM_EPOCHS):
        print('================================================================================================================================')
        print('BEGINNING EPOCH', epoch, ':')
        print('================================================================================================================================')        

        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, config)

        # save model
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

        print("Saving checkpoint ...")
        save_checkpoint(checkpoint)

        # check accuracy
        print("Validating results ...")
        val_loss = validate_fn(val_loader, model, loss_fn, scheduler, train_loss, epoch, time)

        if CONFIG.PROJECT.EARLYSTOP:
            stopping(val_loss, checkpoint, checkpoint_path=PATH+f"/data/checkpoints/{time}_best_checkpoint.pth.tar", epoch = epoch)
            
            if stopping.early_stop:
                print("Early Stopping ...")
                log_predictions(val_loader, model, train_loss, val_loss, epoch, time, device = DEVICE)
                break

    print("Training Finished")