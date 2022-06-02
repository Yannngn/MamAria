import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class PhantomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None, threelabels=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.threelabels=threelabels
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)          

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get Image and corresponding Mask paths
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".tiff", "_mask.png"))

        # Convert Image and Mask to numpy
        image = np.array(Image.open(img_path), dtype=np.float32)
        mask = np.array(Image.open(mask_path), dtype=np.float32)

        # Normalizing Image and Mask pixels to 0 1 interval
        #image = (image / image.max())
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        #Make only 3 labels
        if self.threelabels is True:
            #mask = np.where(mask == 2, 1, mask)
            #mask = np.where(mask == 3, 2, mask)
            mask[mask == 2] = 1
            mask[mask == 3] = 2

        # Applying transforms (ToTensorV2)
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask