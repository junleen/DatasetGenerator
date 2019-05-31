# -*- coding: utf-8 -*-
import glob
import sys
import os
import argparse
import cv2
import filetype
import numpy as np
import face_alignment
from skimage import io
from align import dlibpp
from data.mask import generate_mask, generate_line_mask, generate_region_mask, generate_wider_mask
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
parser.add_argument('--image_size', type=int, default=224, help='Image size (height, width) in pixels.')
parser.add_argument('--detect_multiple_faces', type=bool, default=False, help='Detect and align multiple faces per image.')
parser.add_argument('--num_marks', type=int, default=68, help='The landmarks you want to detect to decide face boundary.')
parser.add_argument('--scale', type=float, default=1.4, help='The factor to scale image')
parser.add_argument('--plus', type=bool, default=True, help='If use cnn to detect face')
parser.add_argument('--use_dlib', type=bool, default=False, help='If use dlib as the landmark detector')
parser.add_argument('--OpenFace', type=str, default='/home/mean/demo/OpenFace/build/bin/FaceLandmarkImg', help='The path of openface you build in')
args = parser.parse_args()
use_dlib = args.use_dlib
use_dlib = False

IMG_EXTENTION = ['.PNG', '.JPG', '.JPEG', '.BMP', '.TIF',
                 '.png', '.jpg', '.jpeg', '.bmp', '.tif']


def join(*args):
    return os.path.join(*args)


def read_imgs(path):
    assert isinstance(path, str)
    file_list = os.listdir(path)
    file_list.sort()
    img_list = []
    for item in file_list:
        path = join(path, item)
        if is_img(path):
            img_list.append(path)

    return img_list


def is_img(path):
    _, suffix = os.path.splitext(path)
    if suffix in IMG_EXTENTION:
        return True
    else:
        return False


# 确定一个最大的人脸
def nms(boxes):
    '''Select a max face to save
    Args:
        boxes: (n, 4) matrix or list like [[x1,y1,x2,y2],[...],...]

    Return:
        box: the front face which close to the camera.
    '''
    area = []
    for box in boxes:
        area.append((box[2]-box[0])*(box[3]-box[1]))

    return area.index(max(area))


# 图片检测与保存
def get_bounding_box(detector, image, conditioned_on_box=False, scale=1.4):
    '''given conditioned box and scale, resize and get a new box
    Args:
        detector: face detector
        image: input image
        conditioned_on_box: generate new box conditioned on box
        scale: rescale factor, default 1.2

    Return:
        box: (4,)
        landmarks: (68, 2)
    '''
    if use_dlib:
        box, landmarks = detector.detect_faces(image)
    else:
        landmarks = detector.get_landmarks(image)
        if isinstance(landmarks, type(None)):
            return None, None
    shape = image.shape
    if use_dlib:
        if len(box) > 1:
            idx = nms(box)
            box, landmarks = box[idx], landmarks[idx]
        elif len(box)==1:
            box, landmarks = box[0], landmarks[0]
    else:
        landmarks = landmarks[0]
        box = generate_crop_box(image_info=landmarks, scale=scale)

    box[0] = 0 if box[0] < 0 else box[0]
    box[1] = 0 if box[1] < 0 else box[1]
    box[2] = shape[1] if box[2] > shape[1] else box[2]
    box[3] = shape[0] if box[3] > shape[0] else box[3]

    return box, landmarks


def generate_crop_box(image_info=None, scale=1.2):
    '''
    giving provided image_info and rescale the box to new size
    Args:
        image_info: the bounding box or the landmarks

    Return:
        a box with 4 values: [left, top, right, bottom] or a
        list contains several box, each has 4 landmarks
    '''
    box = None
    if image_info is not None:
        if np.max(image_info.shape) > 4:  # key points to get bounding box
            kpt = image_info
            if kpt.shape[0] < 3:
                kpt = kpt.T   # nof_marks x 2
            if kpt.shape[0] <= 5:  # 5 x 2
                scale = scale*scale
            left = np.min(kpt[:, 0])
            right = np.max(kpt[:, 0])
            top = np.min(kpt[:, 1])
            bottom = np.max(kpt[:, 1])
        else:  # bounding box
            bbox = image_info
            left = bbox[0]
            right = bbox[2]
            top = bbox[1]
            bottom = bbox[3]

        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * scale)
        box = [center[0] - size / 2, center[1] - size / 2,
               center[0] + size / 2, center[1] + size / 2]

    return box


def save_image(image_fold, image_name, image):

    try:
        if not os.path.exists(image_fold):
            os.mkdir(image_fold)
        save_path = join(image_fold, image_name)
        io.imsave(save_path, image)
        return save_path
    except:
        return None

##################################################
# 读取图片集文件夹，并可以得到所有视频的label idx以及路径
class ImageFolder(object):
    def __init__(self, path):
        self.classes, self.idx_to_class, self.samples = self.__read_fold__(path)

    def __read_fold__(self, path):
        '''Read a fold, and get the classes and its sub videos
        Args:
            path: dataset path

        Return:
            classes: list contain all classes
            idx_to_classes: dict, idx=label index
            samples: list contain all (idx, video path)
        '''
        subdirs = os.listdir(path)
        subdirs.sort()
        idx_to_class = {}
        samples = []
        classes = []
        idx = 0
        for item in subdirs:
            dirpath = os.path.join(path, item)
            if os.path.isdir(dirpath):
                idx_to_class[idx] = item
                classes.append(item)
                samples += self.__read_files__(idx, dirpath)
                idx += 1

        return classes, idx_to_class, samples

    def __read_files__(self, idx, path):
        '''Read a fold, and get the videos. Using filetype to judge the files type
        Args:
            idx: label index
            path: subdir of dataset path
        Return:
            filepaths: list contain all (idx, video path)
        '''
        files = os.listdir(path)
        files.sort()
        filepaths = []
        for file in files:
            _filepath = os.path.join(path, file)
            if os.path.isdir(_filepath):
                filepaths += self.__read_files__(_filepath)
            else:
                filekind = filetype.guess(_filepath)
                if filekind is not None and filekind.mime.split('/')[0] == 'image':
                    filepaths.append((idx, _filepath))

        return filepaths


def fuck_this_image(detector, image_path, save_path, classname, image_size, scale):
    _, file = os.path.split(image_path)
    filename, suffix = os.path.splitext(file)


    image = io.imread(image_path)
    box, landmarks = get_bounding_box(detector, image, False, scale)
    if isinstance(landmarks, type(None)):
        return None
    if len(landmarks) != 0:
        box = np.int32(box)
        line, region, face = generate_mask(image, landmarks)
        face_resize = cv2.resize(image[box[1]:box[3], box[0]:box[2]], (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        line = cv2.resize(line[box[1]:box[3], box[0]:box[2]], (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        region = cv2.resize(region[box[1]:box[3], box[0]:box[2]], (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        face = cv2.resize(face[box[1]:box[3], box[0]:box[2]], (image_size, image_size), interpolation=cv2.INTER_CUBIC)

        # save
        save_img_path = save_image(join(save_path['img'], classname), filename+'_img%s'%suffix, face_resize)
        save_line_path = save_image(join(save_path['line'], classname), filename+'_line%s'%suffix, line)
        save_region_path = save_image(join(save_path['region'], classname), filename+'_region%s'%suffix, region)
        save_face_path = save_image(join(save_path['face'], classname), filename+'_face%s'%suffix, face)

        staturs = (save_img_path is not None) and (save_line_path is not None) and (save_region_path is not None) and (save_face_path is not None)

        if staturs:
            return [save_img_path, save_line_path, save_region_path, save_face_path], (box, landmarks)
        else:
            return None
    return None


def main(args):
    image_size = args.image_size
    scale = args.scale

    imgfolder = ImageFolder(args.input_dir)

    # 定义保存路径
    save_path = {}
    save_path['root'] = args.output_dir
    save_path['img'] = os.path.join(save_path['root'], 'img')
    save_path['line'] = os.path.join(save_path['root'], 'line')
    save_path['region'] = os.path.join(save_path['root'], 'region')
    save_path['face'] = os.path.join(save_path['root'], 'face')

    if not os.path.exists(save_path['root']):
        os.mkdir(save_path['root'])
    if not os.path.exists(save_path['img']):
        os.mkdir(save_path['img'])
    if not os.path.exists(save_path['line']):
        os.mkdir(save_path['line'])
    if not os.path.exists(save_path['region']):
        os.mkdir(save_path['region'])
    if not os.path.exists(save_path['face']):
        os.mkdir(save_path['face'])

    # 写第一行标注
    info_file = open(save_path['root'] + '/infomation.csv', 'w')
    info_file.write('aligned_imgs,line_imgs,region_imgs,face_imgs,')
    info_file.write('x,y,w,h')
    for i in range(68):
        info_file.write(',x_%d,y_%d' % (i, i))
    info_file.write('\n')

    # 定义人脸检测器
    if use_dlib:
        face_detector = dlibpp.DLIBPP(68)
    else:
        face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

    # 遍历整个项目数据集
    for (class_idx, img_path) in tqdm(imgfolder.samples):
        print('Aligning [%s] in [%s]' %(os.path.split(img_path)[1], imgfolder.idx_to_class[class_idx]))
        results = fuck_this_image(face_detector, img_path, save_path, imgfolder.idx_to_class[class_idx], image_size, scale)
        if results is not None:
            paths, info = results[0], results[1]
            for item in paths:
                info_file.write('%s,' % item)
            box = info[0]
            info_file.write('{},{},{},{}'.format(box[0], box[1], box[2]-box[0], box[3]-box[1]))
            for idx in range(len(info[1])):
                info_file.write(',{},{}'.format(info[1][idx, 0], info[1][idx, 1]))
            info_file.write('\n')
            info_file.flush()
    info_file.close()

    # travel the classes, generate aus and files
    for class_name in tqdm(imgfolder.classes):
        ia = join(save_path['img'], class_name)
        op = join(save_path['root'], 'aus', class_name)
        os.system('%s -fdir %s -aus -out_dir %s' % (args.OpenFace, ia, op))
        delete_txts = glob.glob(join(op, '*.txt'))
        for txt in delete_txts:
            os.system('rm %s' % (txt))
    # os.system('python3 code/prepare_au_annotations.py - ia temp_results/aus/ -op temp_results/aus/')


if __name__ == '__main__':
    main(args)