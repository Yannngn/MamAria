import os
import torch
from PIL import Image
from torch._C import dtype
from torch.utils.data import Dataset
import torchvision.transforms as tvtransforms
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader

class PhantomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.mapping = {0: 0, # 0 = no risk
                        1: 1, # 1 = low risk
                        2: 2, # 2 = medium risk
                        3: 3} # 3 = high risk

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("-crop.tiff", "_mask.png"))
        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


if __name__ == "__main__":

    IMAGE_HEIGHT = 256  # 256 originally
    IMAGE_WIDTH = 98  # 98 originally
    train_dir = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/phantom/"
    train_maskdir = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/mask/"
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            )
        ],
    )

    out = np.array(PhantomDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    ).__getitem__(0))

    print(out.shape)