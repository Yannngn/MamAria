import os
from shutil import copyfile, rmtree
import random

INPUT = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/"
TRAIN = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/train/"
VAL = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/val/"

def main(path = INPUT, train = TRAIN, val = VAL, shuffle = False):
    if shuffle:
        shuffle_train_val(path, train, val)
                    
def shuffle_train_val(path, train_dir, val_dir):
    phantom_orig = path + 'phantom/'
    mask_orig = path + 'mask/'
    phantom_train = train_dir + 'phantom/'
    mask_train = train_dir + 'mask/'
    phantom_val = val_dir + 'phantom/'
    mask_val = val_dir + 'mask/'

    make_dirs(phantom_train)
    make_dirs(mask_train)
    make_dirs(phantom_val)
    make_dirs(mask_val)

    img_files = [os.path.join(path, file) for file in os.listdir(phantom_orig) if file.endswith('.tiff')]
    img_names = set([filepath_to_name(img) for img in img_files])
    val_names = set(random.choices(list(img_names), k=11))
    train_names = img_names - val_names
    
    for file in val_names:
        copyfile(phantom_orig + file + '.tiff', phantom_val + file + '.tiff')
        copyfile(mask_orig + file[:-5] + '_mask.png', mask_val + file[:-5] + '_mask.png')

    for file in train_names:
        copyfile(phantom_orig + file + '.tiff', phantom_train + file + '.tiff')
        copyfile(mask_orig + file[:-5] + '_mask.png', mask_train + file[:-5] + '_mask.png')

def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

def make_dirs(path, delete = True):
    if os.path.isdir(path) and (delete == True):
        try:
            rmtree(path)
        except OSError as e:
            raise ValueError("Error: %s : %s" % (path, e.strerror))
        
        os.makedirs(path)

    elif os.path.isdir(path) is not True:
        
        os.makedirs(path)

    else:
        raise ValueError("Directory already exists")
       
if __name__ == "__main__":
    main(shuffle=True)