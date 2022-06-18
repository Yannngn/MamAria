import torch
import wandb

from tqdm import tqdm

from utils.utils import get_device, save_checkpoint
from validate import validate_fn, early_stop_validation

def train_fn(loader, model, optimizer, loss_fn, scaler, config):
    device = get_device(config)
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
    
    wandb.log({"loss":closs/config.hyperparameters.batch_size})
    loop.close()
    return loss.item()

def train_loop(train_loader, val_loader, model, optimizer, scheduler, loss_fn, scaler, stopping, global_metrics, label_metrics, config):  
    for epoch in range(config.epoch, config.project.max_num_epochs):      
        print(f'='.center(125, '='))
        print(f'   BEGINNING EPOCH {epoch}:   '.center(125,'='))

        config.epoch = epoch
        wandb.log({"epoch": epoch})
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, config)
        # check accuracy
        
        print(f'='.center(125, '='))
        print("Validating results ... \n")
        
        val_loss = validate_fn(val_loader, model, loss_fn, scheduler, global_metrics, label_metrics, config)
        
        # save model
        checkpoint = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
                
        save_checkpoint(checkpoint)

        if not config.project.earlystop: continue
        
        stopping(val_loss, checkpoint, checkpoint_path=f"data/checkpoints/{config.time}_best_checkpoint.pth.tar", epoch=epoch)
        
        if stopping.early_stop: 
            early_stop_validation(val_loader, model, global_metrics, label_metrics, config)
            wandb.finish()
            break
    
    wandb.finish()

    print(f'='.center(125, '='))
    print("Training Finished")

    