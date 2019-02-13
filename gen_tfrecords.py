#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import cv2
import tensorflow as tf
from time import time

vh = 800
vw = 2048

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
   
    tf.app.flags.DEFINE_string("output_dir", "tfrecords/", "")
    tf.app.flags.DEFINE_string("cityscapes_root", "/mnt/f3be6b3c-80bb-492a-98bf-4d0d674a51d6/cityscapes/", "")
    #tf.app.flags.DEFINE_string("cityscapes_root", "/home/nate/data/cityscapes", "")
    tf.app.flags.DEFINE_integer("num_files", 7, "Num files to write for train dataset. More files=better randomness")
    tf.app.flags.DEFINE_boolean("debug", False, "")
    

    if FLAGS.debug:
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch
        plt.ion()
        imdata = None

def writeFileList(dirName):
    im_list = [] # list of all files with full path
    lab_list = [] # list of all files with full path
    for dirname, dirnames, filenames in os.walk(os.path.join(dirName, 'gtFine')):
        for filename in filenames:
            if filename.endswith('.png'):	
                fileName = os.path.join(dirname, filename)
                lab_list.append(fileName)
                bnl = fileName.split('gtFine')
                im_list.append(bnl[0] + "leftImg8bit" + \
                        bnl[1] + "leftImg8bit.png")
                
    return im_list, lab_list


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def upper_crop(x, w, h):
    # remove the benz hood from the frame
    j = x.shape[1] // 2
    return x[:h, (j-w//2):(j+w//2), :]

def generate():
   
    if not os.path.isdir(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    train_writers = []
    for ii in range(FLAGS.num_files):
        train_writers.append(None if FLAGS.debug else \
                tf.python_io.TFRecordWriter(FLAGS.output_dir + "train_data%d.tfrecord" % ii))
    val_writer = None if FLAGS.debug else \
            tf.python_io.TFRecordWriter(FLAGS.output_dir + "validation_data.tfrecord")

    car_id = 26

    im_list, lab_list = writeFileList(FLAGS.cityscapes_root)

    count = 1
    for i in range(len(im_list)):
        im_fl = im_list[i]
        lab_fl = lab_list[i]

        print("Working on sample %d" % i)

        image = upper_crop(cv2.imread(im_fl), vw, vh)
        lab = upper_crop(cv2.imread(lab_fl, 
            cv2.IMREAD_GRAYSCALE)[..., np.newaxis], vw, vh)

        mask_label = np.zeros((vh, vw, 2), dtype=np.bool)
        mask_label[:, :, 1:2] = lab==car_id
        if np.any(mask_label[:,:,1]):
            mask_label[:, :, 0] = np.logical_not(mask_label[:, :, 1])
            if FLAGS.debug:
                mask = np.argmax(mask_label, axis=-1)
                rgb = np.zeros((vh, vw, 3))

                legend = []
                np.random.seed(0)
                for i in range(2):
                    c = np.random.rand(3)
                    case = mask==i
                    if np.any(case):
                        legend.append(Patch(facecolor=tuple(c), edgecolor=tuple(c),
                                    label='background' if i==0 else 'car'))

                    rgb[case, :] = c
                
                _image = cv2.resize(image, (vw, vh)) / 255.0

                _image = 0.3 * _image + 0.7 * rgb

                global imdata
                if imdata is None:
                    imdata = plt.imshow(_image)
                    f = plt.gca()
                    f.axes.get_xaxis().set_ticks([])
                    f.axes.get_yaxis().set_ticks([])
                else:
                    imdata.set_data(_image)

                lgd = plt.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.0, 1))
                
                plt.pause(1e-9)
                plt.draw()
                plt.pause(3)

            else:
                features_ = {
                    'img': bytes_feature(tf.compat.as_bytes(image.tostring())),
                    'label': bytes_feature(tf.compat.as_bytes(mask_label.astype(np.uint8).tostring()))
                }
                example = tf.train.Example(features=tf.train.Features(feature=features_))

                if 'val' in im_fl:
                    val_writer.write(example.SerializeToString())
                else:
                    train_writers[np.random.randint(0,FLAGS.num_files)].write(example.SerializeToString())
            count += 1
        else:
            print("No cars. Skipping")
    print("Done. Sample count =", count)
def main(argv):
    del argv
    generate()


if __name__ == "__main__":
    tf.app.run()
