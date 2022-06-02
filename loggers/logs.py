import json
import torch

from datetime import datetime
from munch import munchify
from yaml import safe_load

from utils.save_images import save_predictions_as_imgs, save_submission_as_imgs, save_ellipse_pred_as_imgs, save_confidence_as_imgs
from utils.utils import check_accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))

def log_early_stop(val_loader, model, loss_train, loss_val, epoch, time=0, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=DEVICE):
    print(f'='.center(125, '='))
    print("Early Stopping ...")
    
    if time != 0: time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_predictions(val_loader, model, loss_train, loss_val, epoch, time, folder, device)

def log_predictions(val_loader, model, loss_train, loss_val, epoch, time=0, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=DEVICE):
    print(f'='.center(125, '='))
    print("   Logging and saving predictions...   ".center(125, '='))
    
    dict_eval  = check_accuracy(val_loader, model, device)

    dict_eval['epoch'] = epoch
    dict_eval['loss_train'] = loss_train
    dict_eval['loss_val'] = loss_val
    
    if time != 0: time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_predictions_as_imgs(val_loader, model, epoch, dict_eval, time=time, folder=folder, device=device)
    #save_ellipse_pred_as_imgs(val_loader, model, epoch, dict_eval, time=now, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=device)
    #save_confidence_as_imgs(val_loader, model, epoch, dict_eval, time=now, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=device)

def log_submission(loader, model, loss_test, time=0, folder=CONFIG.PATHS.SUBMISSIONS_DIR, device = DEVICE):
    print(f'='.center(125, '='))
    print("   Logging and saving submissions...   ".center(125, '='))
    
    dict_subm = check_accuracy(loader, model, device)
    dict_subm['loss_subm'] = loss_test
    
    with open(folder+f'{time}_submission.json','w') as f:
        json.dump(dict_subm, f, ensure_ascii=False, indent=4)  

    if time != 0: time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_submission_as_imgs(loader, model, dict_subm, time=time, folder=folder, device=device)
    #save_confidence_as_imgs(loader, model, 0, dict(), time=now, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=device)