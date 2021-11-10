import torch
from tqdm import tqdm
from datetime import datetime
from munch import munchify
from yaml import safe_load

from utils import check_accuracy, print_and_save_results, save_predictions_as_imgs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))

def validate_fn(val_loader, model, loss_fn, scheduler, train_loss, epoch, idx, time):
    loop = tqdm(val_loader)
    model.eval()
    
    for _, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.long().to(DEVICE)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        num_correct, num_pixels, metrics = check_accuracy(val_loader, model, DEVICE)

        if CONFIG.PROJECT.SCHEDULER:
            scheduler.step(loss.item())
        
        print_and_save_results(num_correct, num_pixels, metrics, train_loss, loss.item(), epoch, idx, time=time)
        
        # Print predictions to folder
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_predictions_as_imgs(val_loader, model, epoch, folder = CONFIG.PATHS.PREDICTIONS_DIR, time=now, device = DEVICE)

    model.train()
    
    return loss.item()