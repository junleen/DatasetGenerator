'''
cohn-kanade-images/S005/001/S005_001_00000011.png
------------------------------------------------------------------------------------
will have the corresponding landmark at:

Landmarks/S005/001/S005_001_00000011_landmarks.txt
'''
import cv2
import numpy as np
from PIL import Image


def crop_face(img, landmarks, scale=1.2, minsize=True, dstsize=None):

    xin, yin, xax, yax = generate_bbox_from_landmark(landmarks, scale, minsize)

    if isinstance(img, Image.Image):
        face = img.crop([xin, yin, xax, yax])
    elif isinstance(img, np.ndarray):
        face = img[yin:yax, xin:xax]
    else:
        print('Not desired input image type. Return None')
        face = None
    if face is not None:
        if not dstsize:
            if isinstance(dstsize, tuple):
                face = resize(img, dstsize)
            elif isinstance(dstsize, int):
                face = resize(img, (dstsize, dstsize))

    return face

def generate_bbox_from_landmark(landmarks, scale=1.0, minsize=False):
    xax, yax = np.max(landmarks, axis=0)
    xin, yin = np.min(landmarks, axis=0)
    if minsize:
        size = min(yax - yin, xax - xin)
    else:
        size = max(yax - yin, xax - xin)
    center = (xin + (xax - xin) / 2, yin + (yax - yin) / 2)

    size = scale * size
    xin, xax = center[0] - size / 2, center[0] + size / 2
    yin, yax = center[1] - size / 2, center[1] + size / 2

    return [xin, yin, xax, yax]


def resize(img, dstsize):
    face = None
    if isinstance(img, Image.Image):
        face = img.resize(dstsize)
    elif isinstance(img, np.ndarray):
        face = cv2.resize(img, dstsize,interpolation=cv2.INTER_LINEAR)

    return face

