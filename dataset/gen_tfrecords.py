#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import cv2
import tensorflow as tf
from time import time

from coco_classes import coco_classes, unet_classes
import coco

vw = 640
vh = 480

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
   
    tf.app.flags.DEFINE_string("output_dir", "/mnt/f3be6b3c-80bb-492a-98bf-4d0d674a51d6/coco/unet_tfrecords/", "")
    tf.app.flags.DEFINE_string("coco_root", "/mnt/f3be6b3c-80bb-492a-98bf-4d0d674a51d6/coco/", "")
    tf.app.flags.DEFINE_integer("num_files", 7, "Num files to write for train dataset. More files=better randomness")
    tf.app.flags.DEFINE_boolean("debug", False, "")
    

    if FLAGS.debug:
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch
        plt.ion()
        imdata = None

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate():
    if not os.path.isdir(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)


    train_writers = []
    '''
    for ii in range(FLAGS.num_files):
        train_writers.append(None if FLAGS.debug else \
                tf.python_io.TFRecordWriter(FLAGS.output_dir + "train_data%d.tfrecord" % ii))
    '''
    val_writer = None if FLAGS.debug else \
            tf.python_io.TFRecordWriter(FLAGS.output_dir + "validation_data.tfrecord")

    nclasses = 2
    for split, writer in [('val', val_writer)]:
        # Load dataset
        dataset = coco.CocoDataset()
        dataset.load_coco(FLAGS.coco_root, split)

        # Must call before using the dataset
        dataset.prepare()

        print("Image Count: {}".format(len(dataset.image_ids)))
        print("COCO Class Count: {}".format(dataset.num_classes))
        '''
        for i, info in enumerate(dataset.class_info):
            print("{:3}. {:50}".format(i, info['name']))
        '''


        count = 1
        sample_count = 1
        for image_id in dataset.image_ids:
            print("Working on sample %d" % image_id)

            image = cv2.resize(dataset.load_image(image_id),
                (vw, vh), interpolation=cv2.INTER_CUBIC)
            masks, class_ids = dataset.load_mask(image_id)
            mask_label = np.zeros((vh, vw, nclasses), dtype=np.bool)
            car_cnt = 0
            for i in range(masks.shape[2]):
                cls = coco_classes.get(class_ids[i])
                if cls=='car':
                    car_cnt+=1
                    # Just overwrite because we just want anns with one car
                    mask_label[:, :, 1] = cv2.resize(masks[:, :, i].astype(np.uint8), (vw, vh), 
                            interpolation=cv2.INTER_NEAREST).astype(np.bool)

            print("Car count:", car_cnt)
            # No labels for BG. Make them!
            if car_cnt == 1: 
                mask_label[:, :, 0] = np.logical_not(mask_label[:, :, 1])
                if FLAGS.debug:
                    mask = np.argmax(mask_label, axis=-1)
                    rgb = np.zeros((vh, vw, 3))

                    legend = []
                    np.random.seed(0)
                    for i in range(nclasses):
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

                    if split=='val':
                        writer.write(example.SerializeToString())
                    else:
                        writer[np.random.randint(0,FLAGS.num_files)].write(example.SerializeToString())
                sample_count += 1    
            count += 1
        print("Done. Sample count =", sample_count)
def main(argv):
    del argv
    generate()


if __name__ == "__main__":
    tf.app.run()
