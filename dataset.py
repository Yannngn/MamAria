import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

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

        # Normalizing Image and Mask pixels to 0 1 interval
        image = (image / image.max())

        # Applying transforms (ToTensorV2)
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask