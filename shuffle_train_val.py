import os
import glob
from shutil import copyfile, rmtree
import numpy as np
import random

INPUT = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/"
TRAIN = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/train/"
VAL = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/val/"

def main(path = INPUT, train = TRAIN, val = VAL, shuffle = False):
    if shuffle:
        shuffle_train_val(path, train, val)
                    
def shuffle_train_val(path, train_dir, val_dir):

    make_dirs(train_dir + 'phantom/')
    make_dirs(train_dir + 'mask/')
    make_dirs(val_dir + 'phantom/')
    make_dirs(val_dir + 'mask/')

    img_files = [os.path.join(path, file) for file in os.listdir(path+'phantom/') if file.endswith('.tiff')]
    img_names = set([filepath_to_name(img) for img in img_files])
    val_names = set(random.choices(list(img_names), k=5))
    train_names = img_names - val_names
    
    for file in val_names:
        copyfile(path + 'phantom/' + file + '.tiff', val_dir + 'phantom/' + file + '.tiff')
        copyfile(path + 'mask/' + file[:-5] + '_mask.png', val_dir + 'mask/' + file[:-5] + '_mask.png')

    for file in train_names:
        copyfile(path + 'phantom/' + file + '.tiff', train_dir + 'phantom/' + file + '.tiff')
        copyfile(path + 'mask/' + file[:-5] + '_mask.png', train_dir + 'mask/' + file[:-5] + '_mask.png')

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