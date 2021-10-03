import cv2
import numpy as np
from PIL import Image
import torch

img_path = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/phantom/Richards_888076.1.555465657049.20210203023509248-crop.tiff"
mask_path = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/mask/Richards_888076.1.555465657049.20210203023509248_mask.png"

msk = cv2.imread(mask_path, 0)
#print(np.unique(msk))

img = cv2.imread(img_path, 0)
#print(np.unique(img))

image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
mask = np.array(Image.open(mask_path))

mask_tensor = torch.from_numpy(np.array(mask, dtype=np.uint8))
mask_class = mask_tensor.view(mask.shape[0], mask.shape[1], 1).expand(-1, -1, 3)
print(np.unique(mask_class.shape, mask.shape))