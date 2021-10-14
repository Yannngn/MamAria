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
        # Get Image and corresponding Mask paths
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("-crop.tiff", "_mask.png"))

        # Convert Image and Mask to numpy
        image = np.array(Image.open(img_path), dtype=np.float32)
        mask = np.array(Image.open(mask_path), dtype=np.float32)
        
        # print("IMAGE np.array:", image.shape, image.max(), image.min(), image.mean())
        # print("MASK np.array:", mask.shape, mask.max(), mask.min(), mask.mean())

        # Seting Image and Mask pixels to 0 1 interval
        image = (image / image.max())
        mask = mask / 3

        # print("IMAGE 0 1 interval:", image.shape, image.max(), image.min(), image.mean())
        # print("MASK 0 1 interval:", mask.shape, mask.max(), mask.min(), mask.mean())

        # Applying transforms (Normalize and ToTensorV2)
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # If not applyed Image and Mask outputs will be Numpy and not Tensor

        #print("IMAGE END DATASET:", image.shape, image.max(), image.min(), image.mean())
        #print("MASK END DATASET:", mask.shape, mask.max(), mask.min(), mask.mean())

        return image, mask