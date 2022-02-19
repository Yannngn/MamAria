import cv2
import numpy as np
from sklearn.compose import TransformedTargetRegressor
import torch
import torchvision.transforms as transforms

from munch import munchify
from yaml import safe_load

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))

import matplotlib.pyplot as plt

def label_to_pixel(preds, col='l'):
    if col == 'l':
        preds = preds / (CONFIG.IMAGE.MASK_LABELS - 1) #0, 1, 2 
        preds = preds.unsqueeze(1).float()
        return preds

    else:
        preds = preds[:,1:,:,:]
        return preds.float()

def fit_ellipses_on_image(image: torch.tensor) -> np.array:
    lesion = (CONFIG.IMAGE.MASK_LABELS - 1)
    
    image = image.cpu().numpy()
    image = np.uint8(image)
    original = image.copy()

    for i, img in enumerate(original):
        _, thresh = cv2.threshold(src=img, thresh=lesion-1, maxval=lesion, type=cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        threshed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
       
        contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for cnts in contours:
            convex = cv2.convexHull(cnts)
            try:
                cv2.drawContours(img, [convex], -1, lesion, -1)
            except:
                print('Couldn\'t countour it')
        image[i] = img

    return np.asarray(image, dtype=np.uint8)

def get_confidence_of_prediction(probs: torch.tensor) -> np.array:
    lesion = (CONFIG.IMAGE.MASK_LABELS - 1)
    
    preds = torch.argmax(probs, 1).cpu().numpy()
    preds[preds > 0] = 1
    breast = np.stack((preds,)*3, axis=-1)

    #print(f'original max={torch.max(probs)}, min={torch.min(probs)}, mean={torch.mean(probs)}')
    probs = torch.softmax(probs, 1)
    #print(f'softmax max={torch.max(probs)}, min={torch.min(probs)}, mean={torch.mean(probs)}')

    lesion_prob = probs[:, lesion]
    grayscale = np.array(lesion_prob.cpu() * 255, dtype = np.uint8)
    color = np.zeros((grayscale.shape[0], grayscale.shape[1], grayscale.shape[2], 3))

    for i, img in enumerate(grayscale):
        heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        color[i] = heatmap * breast[i]
    
    lesion_prob = torch.tensor(color).float().to(DEVICE) / 255
    lesion_prob = lesion_prob.permute(0, 3, 1, 2)

    print(torch.max(lesion_prob), torch.min(lesion_prob), lesion_prob.shape)

    return lesion_prob