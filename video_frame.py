# -*- coding: utf-8 -*-

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
parser.add_argument('--image_size', type=int,
                    help='Image size (height, width) in pixels.', default=160)
parser.add_argument('--detect_multiple_faces', type=bool,
                    help='Detect and align multiple faces per image.', default=False)
parser.add_argument('--num_marks', type=int, help='The landmarks you want to detect to decide face boundary.', default=68)
parser.add_argument('--fixed_box', type=bool, help='If use a fixed box to crop face', default=False)
parser.add_argument('--scale', type=float, help='The factor to scale image', default=1.5)
parser.add_argument('--plus', type=bool, help='If use cnn to detect face', default=True)
parser.add_argument('--use_dlib', type=bool, help='If use dlib as the landmark detector', default=False)
args = parser.parse_args()
use_dlib = args.use_dlib


##################################################
# DONE: 将视频抽帧，以第一帧的检测结果为固定的窗口，裁剪视频
##################################################
# 读取视频文件夹，并可以得到所有视频的label idx以及路径
class VideoFolder(object):
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
                if filekind is not None and filekind.mime.split('/')[0] == 'video':
                    filepaths.append((idx, _filepath))

        return filepaths

# 视频读写
class Video(object):
    def __init__(self, video_path):
        super(Video, self).__init__()
        self.cap = self.read_video(video_path)

        # get full parameters
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    # read video
    def read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            return cap
        else:
            print('Cannot read %s' % (video_path))
            return None

    def get_current_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, frame

    def get_frame_by_frameidx(self, frame_idx=None):
        '''
        read a frame from videocap, if assigned frame_idx, then read idxth frame
        Args:
            videocap: opened video cap
            frame_idx: idxth frame to read

        '''
        if isinstance(frame_idx, int) or isinstance(frame_idx, float):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        return self.get_current_frame()

    def get_frame_by_time(self, time_idx=None):
        '''
        Read from at time of time_idx(second)
        Args:
            time_idx: float time

        Return:
            ret, frame
        '''
        if time_idx is not None:
            frame_idx = time_idx * self.fps
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        return self.get_current_frame()


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



def video2images(video_path, save_path, frame_idx, image_size, num_marks=68, fixed_box=False, detect_plus=True, conditioned_box=False, scale=1.4):
    '''Get each fram from a video, and crop it into a fixed size. Mainly holding
    face region
    Args:
        video_path: input video
        save_path: the path to save frames in
        show: show the cropped window or not

    Return:
        None
    '''
    # face_detector = dlibpp.DLIBPP(num_of_marks=num_marks, plus=detect_plus)
    face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

    video = Video(video_path)

    # 读取视频的名称等信息
    dir, file = os.path.split(video_path)
    class_name = os.path.split(dir)[1]
    video_name = os.path.splitext(file)[0]

    # 定义保存路径
    save_frame_path = os.path.join(save_path, 'imgs')
    save_line_path = os.path.join(save_path, 'line')
    save_region_path = os.path.join(save_path, 'region')
    save_face_path = os.path.join(save_path, 'face')

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_frame_path):
        os.mkdir(save_frame_path)
    if not os.path.exists(save_line_path):
        os.mkdir(save_line_path)
    if not os.path.exists(save_region_path):
        os.mkdir(save_region_path)
    if not os.path.exists(save_face_path):
        os.mkdir(save_face_path)

    # read video frame
    ret, frame = video.get_current_frame()
    box, landmarks = get_bounding_box(face_detector, frame, conditioned_box, scale)
    frame_idx = frame_idx
    while ret!=False and isinstance(box, type(None)):
        ret, frame = video.get_current_frame()
        box, landmarks = get_bounding_box(face_detector, frame, conditioned_box, scale)
        frame_idx += 1
    box = np.int32(box)
    _temp_box = box.copy()

    while ret:
        current_frame = os.path.join(save_frame_path, '%s_img_%.5d.png' % (class_name, frame_idx))
        if os.path.exists(current_frame):
            ret, frame = video.get_current_frame()
            frame_idx += 1
            continue

        if fixed_box:
            box, landmarks = get_bounding_box(face_detector, frame, conditioned_box, scale)
            box = _temp_box
        else:
            box, landmarks = get_bounding_box(face_detector, frame, conditioned_box, scale)

        if isinstance(landmarks, type(None)):
            ret, frame = video.get_current_frame()
            frame_idx += 1
            continue
        if len(landmarks) != 0:
            box = np.int32(box)
            # print(box, landmarks)
            line, region, face = generate_mask(frame, landmarks)

            face_resize = cv2.resize(frame[box[1]:box[3], box[0]:box[2]], (image_size, image_size),interpolation=cv2.INTER_CUBIC)
            line = cv2.resize(line[box[1]:box[3], box[0]:box[2]], (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            region = cv2.resize(region[box[1]:box[3], box[0]:box[2]], (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            face = cv2.resize(face[box[1]:box[3], box[0]:box[2]], (image_size, image_size), interpolation=cv2.INTER_CUBIC)

            # save
            io.imsave(os.path.join(save_frame_path, '%s_img_%.5d.png' % (class_name, frame_idx)), face_resize)
            io.imsave(os.path.join(save_line_path, '%s_line_%.5d.png' % (class_name, frame_idx)), line)
            io.imsave(os.path.join(save_region_path, '%s_region_%.5d.png' % (class_name, frame_idx)), region)
            io.imsave(os.path.join(save_face_path, '%s_face_%.5d.png' % (class_name, frame_idx)), face)

            #for (x,y) in landmarks:
            #    frame = cv2.circle(frame, (x,y), 1, (255,255,255),1)

            #cv2.imshow(class_name, frame);cv2.waitKey(10)
        ret, frame = video.get_current_frame()
        frame_idx += 1
    # if show:
    # cv2.destroyAllWindows()
    return frame_idx

# main function
def main(args):
    print(args)
    videofolder = VideoFolder(args.input_dir)
    conditioned_box = False
    fixed_box = False
    print('fixed_box:', fixed_box)
    print('use_dlib', use_dlib)

    idx_logger = dict()
    for key in videofolder.classes:
        idx_logger[key] = 0


    for (class_idx, video_path) in tqdm(videofolder.samples):
        print('Aligning [%s] in [%s]' %(os.path.split(video_path)[1],\
                                        videofolder.idx_to_class[class_idx]))

        classname = videofolder.idx_to_class[class_idx]
        frame_idx = video2images(video_path=video_path, save_path=args.output_dir, frame_idx=idx_logger[classname], image_size=args.image_size, num_marks=args.num_marks, fixed_box=fixed_box, detect_plus=args.plus, conditioned_box=conditioned_box, scale=args.scale)
        idx_logger[classname] = frame_idx

if __name__ == '__main__':
    main(args)

#TODO: OpenFace Action Unit, use FeatureExtraction

