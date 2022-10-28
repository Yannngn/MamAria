import logging
import torch
import wandb

from tqdm import tqdm

from utils.utils import get_device, save_checkpoint
from validate import validate_fn, early_stop_validation

def train_fn(loader, model, optimizer, loss_fn, scaler, config):
    device = get_device(config)
    logging.info('Training model...')
    
    loop = tqdm(loader, bar_format='{l_bar}{bar:75}{r_bar}{bar:-75b}')
    closs = 0.

    model.train()
    for data, targets in loop:
        data, targets = data.to(device), targets.long().to(device)

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
    
    wandb.log({"loss":closs/config.hyperparameters.batch_size})
    loop.close()
    
    return loss.item()

def train_loop(train_loader, 
               val_loader, 
               model, 
               optimizer, 
               scheduler, 
               loss_fn, 
               scaler, 
               stopping, 
               global_metrics, 
               label_metrics, 
               config
               ):  
    
    for epoch in range(config.project.epoch, config.project.num_epochs):      
        logging.info(f'Starting epoch {epoch}...')

        config.project.epoch = epoch
        wandb.log({"epoch": epoch})
        
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, config)
        
        # check accuracy
        val_loss = validate_fn(val_loader, model, loss_fn, scheduler, global_metrics, label_metrics, config)
        
        # save model
        logging.info('Saving trained weights...')
        
        checkpoint = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
                
        save_checkpoint(checkpoint)
        
        stopping(val_loss, checkpoint, checkpoint_path=f"data/checkpoints/{config.project.time}_best_checkpoint.pth.tar", epoch=epoch)
        
        if not stopping.early_stop: continue
            
        early_stop_validation(val_loader, model, global_metrics, label_metrics, config)

        break
    
    wandb.finish()

    logging.info('Training finished...')

    