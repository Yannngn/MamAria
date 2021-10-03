import os
import torch
from PIL import Image
from torch._C import dtype
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np
import albumentations as A

class PhantomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        self.mapping = {(0, 0, 0): 0, # 0 = no risk / background
                        (1, 1, 1): 1, # 1 = low risk
                        (2 ,2, 2): 2, # 2 = medium risk
                        (3, 3, 3): 3} # 3 = high risk
        
    def mask_to_class(self, mask):
        mask_class = mask.view(mask.shape[0], mask.shape[1], 1).expand(-1, -1, 3)
        for k in self.mapping:
            for mask[mask == k] = self.mapping[k]
        return mask
    
    def __getitem__(self, index):

        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("-crop.tiff", "_mask.png"))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)) # this is for my dataset(lv)
        mask = self.mask_to_class(mask)
        mask = mask.float()

        print(mask.shape)
        print(np.unique(mask))

        return image, mask
    
    def __len__(self):  # return count of sample we have
        return len(self.images)