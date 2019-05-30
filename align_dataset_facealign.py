'''
Use dlib to detect and align faces
'''
from scipy import misc
import sys
import os
import argparse
import cv2
import tensorflow as tf
import numpy as np
import face_alignment
import facenet
import random
from time import sleep
from data.mask import generate_mask
from data.align import generate_bbox_from_landmark



def main(args):
    print(args)
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)

    print('Creating networks and loading parameters')
    face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=False)

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, 'imgs', cls.name)
            output_class_mask_dir = os.path.join(output_dir, 'masks', cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            if not os.path.exists(output_class_mask_dir):
                os.makedirs(output_class_mask_dir)

            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')
                output_maskname = os.path.join(output_class_mask_dir, filename+'.png')

                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:, :, 0:3]

                        points = face_detector.get_landmarks(img)


                        nrof_faces = len(points)
                        if nrof_faces > 0:
                            bounding_boxes = []
                            for landmark in points:
                                bounding_boxes.append(generate_bbox_from_landmark(landmark))
                            bounding_boxes = np.int32(bounding_boxes)
                            mask = generate_mask(img, points)

                            det = bounding_boxes[:, 0:4]
                            det_arr = []
                            landmarks_arr = []
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces > 1:
                                if args.detect_multiple_faces:
                                    for i in range(nrof_faces):
                                        det_arr.append(np.squeeze(det[i]))
                                        landmarks_arr.append(points[i])
                                else:
                                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                    index = np.argmax(
                                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                    det_arr.append(det[index, :])
                                    landmarks_arr.append(points[index])
                            else:
                                det_arr.append(np.squeeze(det))

                            for i, det in enumerate(det_arr):
                                det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
                                old_size = (det[2] - det[0] + det[3] - det[1]) / 2.0
                                center = np.array([det[2] - (det[2] - det[0]) / 2.0,
                                                   det[3] - (det[3] - det[1]) / 2.0])
                                size = int(old_size * args.zoom)
                                det[0], det[1], det[2], det[3] = center[0] - size / 2, center[1] - size / 2 + 10, center[
                                    0] + size / 2, center[1] + size / 2 + 10

                                det[0] = 0 if det[0] < 0 else det[0]
                                det[1] = 0 if det[1] < 0 else det[1]
                                det[2] = det[2] if det[2] < img.shape[1] else img.shape[1]
                                det[3] = det[3] if det[3] < img.shape[1] else img.shape[1]

                                # cropped = dlibpp_detector.__align_rotate__(img, det, points[:, i])
                                # print('Shape', cropped.shape, det)
                                cropped = img[det[1]:det[3], det[0]:det[2]]
                                cropped_mask = mask[det[1]:det[3], det[0]:det[2]]

                                scaled = cv2.resize(cropped, (args.image_size, args.image_size), cv2.INTER_LINEAR)
                                scaled_mask = cv2.resize(cropped_mask, (args.image_size, args.image_size), cv2.INTER_LINEAR)
                                #scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')

                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)
                                maskname_base, mask_extention = os.path.splitext(output_maskname)

                                if args.detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                    output_maskname_n = "{}_{}{}".format(maskname_base, i, mask_extention)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                    output_maskname_n = "{}{}".format(maskname_base, mask_extention)
                                misc.imsave(output_filename_n, scaled)
                                misc.imsave(output_maskname_n, scaled_mask)
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=20)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.4)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    parser.add_argument('--num_marks', type=int, help='The landmarks you want to detect to decide face boundary.', default=68)
    parser.add_argument('--scale', type=float, help='The factor to scale image', default=1.4)
    parser.add_argument('--plus', type=bool, help='If use a better detector, or get more frontal face', default=True)
    parser.add_argument('--zoom', type=float, default=1.2)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
