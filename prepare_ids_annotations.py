import numpy as np
import os
from tqdm import tqdm
import argparse
import glob
import re
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', type=str, help='Dir with imgs aus files')
parser.add_argument('-out_name', type=str, help='Output path', default='ids')
args = parser.parse_args()


def join(*path):
    return os.path.join(*path)

def get_classes(path):
    classes = os.listdir(path)
    classes.sort()
    return classes

def get_class_imgs(path):
    filepaths = os.listdir(path)
    filepaths.sort()
    #relative_path = [join(path, filepath) for filepath in filepaths]
    return filepaths

def main(args):
    aus_path = join(args.data_dir, 'aus_npy')
    img_path = join(args.data_dir, 'imgs')
    mask_path = join(args.data_dir, 'mask')

    classes = get_classes(img_path)
    stack_aus = []
    with open(args.out_name+'.csv', 'w') as ids:
        for class_i in classes:
            class_imgpaths = get_class_imgs(join(img_path, class_i))
            class_maskpaths = get_class_imgs(join(mask_path, class_i))
            class_aus = np.load(join(aus_path, class_i+'.npy'))

            for idx_i in range(class_aus.shape[0]):
                class_imgpath = class_imgpaths[idx_i]
                class_maskpath = class_maskpaths[idx_i]
                class_au = class_aus[idx_i, 1:]
                ids.write("%s,%s\n" %(join('imgs', class_i, class_imgpath), join('mask/', class_i, class_maskpath)))
                stack_aus.append(class_au)
            ids.flush()
        stack_aus = np.array(stack_aus)
        np.save(args.out_name+'.npy', stack_aus)

if __name__ == "__main__":
    main(args)