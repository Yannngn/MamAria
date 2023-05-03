import os
import random
from shutil import copyfile, rmtree

import numpy as np

# PATH = '/'.join(os.path.abspath(__file__).split('\\')[:-2])
# + "/LesionInserter-data/"
PATH = "E:/giulia_folder/LesionInserter-master-perlin-new/"
OUT = "data/"

IMG_PATH = OUT + "phantom/"
MSK_PATH = OUT + "mask/"
TRAIN = OUT + "train/"
VAL = OUT + "val/"
TEST = OUT + "test/"

FILES = [
    "sphere_0_2mass-raw",
    # 'spiculated_1_2mass',
    # 'spiculated_2_2mass',
    # 'spiculated_3_2mass',
]


def main(
    path=PATH,
    files=FILES,
    img_path=IMG_PATH,
    mask_path=MSK_PATH,
    get_slice=False,
    delete=False,
    shuffle=False,
):
    if delete:
        clear_data(img_path)
        clear_data(mask_path)
        clear_data(TRAIN)
        clear_data(VAL)
        clear_data(TEST)

    if get_slice:
        for file in files:
            get_slices(path + file + "/", img_path, mask_path)

    if shuffle:
        shuffle_train_val(OUT, TRAIN, VAL, TEST)


# original = phantom_55_10.9_per_1_lac_2_idx_25-crop
# name = 109_per_1_lac_2_idx_25-crop
# final = 55_109_1_2_25


def get_slices(path, img_path=IMG_PATH, mask_path=MSK_PATH):
    lesion = os.path.basename(os.path.normpath(path))
    for _, dirs, _ in os.walk(path):
        for folder in dirs:
            if folder[-4:] == "crop":
                slice_number = int(len(os.listdir(path + folder)) * 0.5)
                name = rename_image("_".join(folder.split("_")[1:])[:-5])
                copyfile(
                    path + folder + f"/_recon_slice_{slice_number}.tiff",
                    img_path + lesion + "_" + "_".join(name.split("_")) + ".tiff",
                )

                copyfile(
                    path + folder[:-5] + f"_mask/_{slice_number}.png",
                    mask_path + lesion + "_" + "_".join(name.split("_")) + "_mask.png",
                )


def rename_image(name):
    lst = name.split("_")
    temp = lst[1].split(".")
    lst[1] = f"{int(temp[0]):02}" + f"{int(temp[1])*10}"[:2]
    if lst[3] == "1":
        lst[3] = "100"
    if lst[3] == "0.75":
        lst[3] = "075"

    return lst[0] + "_" + "_".join(lst[1::2])


def clear_data(path):
    if os.path.isdir(path):
        try:
            rmtree(path)
        except OSError as e:
            raise ValueError("Error: %s : %s" % (path, e.strerror))

        os.makedirs(path)

    elif os.path.isdir(path) is False:
        try:
            os.makedirs(path)
        except OSError as e:
            raise ValueError("Error: %s : %s" % (path, e.strerror))


def shuffle_train_val(path, train_dir, val_dir, test_dir):
    phantom_orig = path + "phantom/"
    mask_orig = path + "mask/"
    phantom_train = train_dir + "phantom/"
    mask_train = train_dir + "mask/"
    phantom_val = val_dir + "phantom/"
    mask_val = val_dir + "mask/"
    phantom_test = test_dir + "phantom/"
    mask_test = test_dir + "mask/"

    os.makedirs(phantom_train)
    os.makedirs(mask_train)
    os.makedirs(phantom_val)
    os.makedirs(mask_val)
    os.makedirs(phantom_test)
    os.makedirs(mask_test)

    img_files = [os.path.join(path, file) for file in os.listdir(phantom_orig) if file.endswith(".tiff")]
    img_names = [filepath_to_name(img) for img in img_files]

    names_per_index = []

    phantoms = np.unique(["_".join(name.split("_")[3:]) for name in img_names])
    print(img_files)

    for phantom in phantoms:
        names = [img for img in img_names if (img.split("_")[3:] == phantom.split("_"))]
        names_per_index.append(names)
    random.shuffle(names_per_index)

    a = len(names_per_index)
    c = int(a * 0.7)
    d = int(a * 0.2)
    e = a - c - d

    train_sort = [0] * a

    train_names = [names[train_sort[n]] for n, names in enumerate(names_per_index[:c])]
    val_names = [names[train_sort[n]] for n, names in enumerate(names_per_index[c:-e])]
    test_names = [names[train_sort[n]] for n, names in enumerate(names_per_index[-e:])]

    for file in test_names:
        copyfile(phantom_orig + file + ".tiff", phantom_test + file + ".tiff")
        copyfile(mask_orig + file + "_mask.png", mask_test + file + "_mask.png")

    for file in val_names:
        copyfile(phantom_orig + file + ".tiff", phantom_val + file + ".tiff")
        copyfile(mask_orig + file + "_mask.png", mask_val + file + "_mask.png")

    for file in train_names:
        copyfile(phantom_orig + file + ".tiff", phantom_train + file + ".tiff")
        copyfile(mask_orig + file + "_mask.png", mask_train + file + "_mask.png")


def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name


if __name__ == "__main__":
    main(get_slice=True, delete=True, shuffle=True)
