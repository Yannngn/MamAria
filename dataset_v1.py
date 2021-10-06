import os
import torch
import imageio
from PIL import Image
from torch._C import dtype
from torch.utils.data import Dataset
from torchvision.utils import save_image
import torchvision.transforms as tvtransforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PhantomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)          

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("-crop.tiff", "_mask.png"))

        # image = imageio.imread(img_path)    
        # image = np.array(Image.fromarray(image), dtype=np.float32)
        image = np.array(Image.open(img_path), dtype=np.float32)
        image = (image / (2 ** 14))
        #print(np.unique(image), image.max())
        mask = np.array(Image.open(mask_path), dtype=np.float32)
        mask[mask == 1.0] = 1/3
        mask[mask == 2.0] = 2/3
        mask[mask == 3.0] = 1.0
        #print(np.unique(mask), mask.max())

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        save_image(image, "teste.tiff")
        save_image(mask, "mask_teste.tiff")

        return image, mask