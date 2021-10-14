import os
import numpy as np
import imageio


def get_weights(mask_dir, labels = 4):
    weights = np.zeros(labels)
    total_pixels = 0
    
    for root, dirs, files in os.walk(mask_dir):
        for filename in files:
            mask = imageio.imread(os.path.join(root, filename))
            
            if total_pixels == 0:
                total_pixels = mask.shape[0] * mask.shape[1]
            
            temp = []
            for i in range(labels):
                
                temp.append((mask == i).sum())

            weights += temp

    return 1 / (weights / total_pixels / len(files))

if __name__ == "__main__":
    weights = get_weights("C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/mask/")
    print(weights, weights.shape, weights.sum())