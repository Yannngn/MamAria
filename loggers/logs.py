#import json

#from datetime import datetime

from utils.save_images import save_predictions_as_imgs, save_confidence_as_imgs, save_calib
from utils.metrics import check_accuracy
#from utils.utils import get_device

def log_predictions(data, label, predictions, global_metrics, label_metrics, config, step):
       
    dict_eval = check_accuracy(predictions, label, global_metrics, label_metrics, config)

    #dict_eval['epoch'] = epoch
    #dict_eval['loss_train'] = loss_train
    #dict_eval['loss_val'] = loss_val
    
    save_predictions_as_imgs(data, label, predictions, config, step, dict_eval)
    #save_ellipse_pred_as_imgs(val_loader, model, epoch, dict_eval, time=now, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=device)
    #save_confidence_as_imgs(predictions, name, config)

def log_calib(label, predictions, global_metrics, label_metrics, config, model, device):

    pre, pos = save_calib(model, predictions, config)
    #results = results.to(device)

    dict_pre_calib  = check_accuracy(pre, label, global_metrics, label_metrics, config)
    dict_pos_calib  = check_accuracy(pos, label, global_metrics, label_metrics, config)

    save_confidence_as_imgs(pre, "pre", config)
    save_confidence_as_imgs(pos, "pos", config)
    
    with open('dict_pre_calib.csv', 'w') as f:
        for key in dict_pre_calib.keys():
            f.write("%s,%s\n"%(key,dict_pre_calib[key]))  
              
    with open('dict_pos_calib.csv', 'w') as f:
        for key in dict_pos_calib.keys():
            f.write("%s,%s\n"%(key,dict_pos_calib[key]))

    return dict_pos_calib, dict_pre_calib

'''def log_submission(predictions, targets, model, loss_test, config, time=0):
    device = get_device(config)
    
    print(f'='.center(125, '='))
    print("   Logging and saving submissions...   ".center(125, '='))
    
    dict_subm = check_accuracy(predictions, targets, model, device)
    dict_subm['loss_subm'] = loss_test
    
    with open(config.paths.submission_dir + f'{time}_submission.json','w') as f:
        json.dump(dict_subm, f, ensure_ascii=False, indent=4)  

    if time == 0: time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_submission_as_imgs(predictions, targets, model, dict_subm, config, time=time)'''
    #save_confidence_as_imgs(loader, model, 0, dict(), time=now, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=device)