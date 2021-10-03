import os
import torch
from PIL import Image
from torch._C import dtype
from torch.utils.data import Dataset
import torchvision.transforms as tvtransforms
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

    def __len__(self):
        return len(self.images)

    def mask_to_class_rgb(self, mask):
            #print('----mask->rgb----')
            mask = torch.from_numpy(mask)
            mask = torch.squeeze(mask)  # remove 1

            # check the present values in the mask, 0 and 255 in my case
            #print('unique values rgb    ', torch.unique(mask)) 
            # -> unique values rgb     tensor([  0, 255], dtype=torch.uint8)

            class_mask = mask
            class_mask = class_mask.permute(2, 0, 1).contiguous()
            h, w = class_mask.shape[1], class_mask.shape[2]
            mask_out = torch.empty(h, w, dtype=torch.float)

            for k in self.mapping:
                idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
                        
                validx = (idx.sum(0) == 3)          
                mask_out[validx] = torch.tensor(self.mapping[k], dtype=torch.float)

            # check the present values after mapping, in my case 0, 1, 2, 3
            #print('unique values mapped ', torch.unique(mask_out))
            # -> unique values mapped  tensor([0, 1, 2, 3])
        
            return mask_out

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("-crop.tiff", "_mask.png"))
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        img_new, mask_new = image, mask

        if self.transform is not None:
            image_np = np.array(image)
            mask_np = np.array(mask, dtype=np.float32) 
            augmentations = self.transform(image = image_np, mask = mask_np)
            img_new = augmentations["image"]
            mask_new = augmentations["mask"]
            img_new = torch.from_numpy(img_new).float()
        else:
            img_new = torch.from_numpy(img_new).float()

        img_new = img_new.permute(2, 0, 1).contiguous()
#       norm = tvtransforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
#       img_new = norm(img_new)

        mask_new = self.mask_to_class_rgb(mask_new)
        mask_new = mask_new.float()

        return img_new, mask_new
