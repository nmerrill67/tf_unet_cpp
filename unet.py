#!/usr/bin/env python3

import os
import sys
import datetime
import tensorflow as tf
from tensorflow.contrib import slim

import numpy as np

from time import time

import utils

from gen_tfrecords import vw as __vw
from gen_tfrecords import vh as __vh

N_CLASSES = 2
vh = 240
vw = 320

FLAGS = tf.app.flags.FLAGS
if __name__ == '__main__':
    tf.app.flags.DEFINE_string("mode", "train", "train or predict")

    tf.app.flags.DEFINE_string("model_dir", "model", "Estimator model_dir")

    tf.app.flags.DEFINE_integer("steps", 50000, "Training steps")
    tf.app.flags.DEFINE_string(
        "hparams", "",
        "A comma-separated list of `name=value` hyperparameter values. This flag "
        "is used to override hyperparameter settings when manually "
        "selecting hyperparameters.")

    tf.app.flags.DEFINE_integer("batch_size", 22, "Size of mini-batch.")
    tf.app.flags.DEFINE_string("input_dir", "tfrecords/", "tfrecords dir")
    tf.app.flags.DEFINE_string("image", "", "Image to predict on")

def create_input_fn(split, batch_size):
    """Returns input_fn for tf.estimator.Estimator.

    Reads tfrecord file and constructs input_fn for training

    Args:
    tfrecord: the .tfrecord file
    batch_size: The batch size!

    Returns:
    input_fn for tf.estimator.Estimator.

    Raises:
    IOError: If test.txt or dev.txt are not found.
    """

    def input_fn():
        """input_fn for tf.estimator.Estimator."""
        
        indir = FLAGS.input_dir
        #tfrecord = 'train_data*.tfrecord' if split=='train' else 'validation_data.tfrecord'
        tfrecord = 'validation_data.tfrecord'

        def parser(serialized_example):


            features_ = {}
            features_['img'] = tf.FixedLenFeature([], tf.string)
            features_['label'] = tf.FixedLenFeature([], tf.string)

            fs = tf.parse_single_example(
                serialized_example,
                features=features_
            )
            
            #if split=='train':
            #    fs['img'] = tf.reshape(tf.cast(tf.decode_raw(fs['img'], tf.uint8),
            #        tf.float32) / 255.0, [__vh,__vw,3])
            #    fs['label'] = tf.reshape(tf.cast(tf.decode_raw(fs['label'], tf.uint8),
            #        tf.float32), [__vh,__vw,N_CLASSES])
            #else:
            fs['img'] = tf.reshape(tf.cast(tf.decode_raw(fs['img'], tf.uint8),
                    tf.float32) / 255.0, [2*vh,2*vw,3])
            fs['label'] = tf.reshape(tf.cast(tf.decode_raw(fs['label'], tf.uint8),
                    tf.float32), [2*vh,2*vw,N_CLASSES])
            return fs

        #if split=='train':
        #    files = tf.data.Dataset.list_files(indir + tfrecord, shuffle=True,
        #            seed=np.int64(time()))
        #else:
        files = [indir + tfrecord]
            
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(400, seed=np.int64(time())))
        dataset = dataset.apply(tf.data.experimental.map_and_batch(parser, batch_size,
                    num_parallel_calls=2))
        dataset = dataset.prefetch(buffer_size=2)

        return dataset

    return input_fn


def unet(images, is_training=False):
    
    # Variational Semantic Segmentator
    with tf.variable_scope("UNet"): 
        images = tf.identity(images, name='images')
        with slim.arg_scope(
            [slim.conv2d],
            normalizer_fn=None,
            activation_fn=lambda x: tf.nn.relu(x),
            padding='SAME'):
            
            ### Encoder ####################################

            d11 = slim.conv2d(images, 64, [3,3])
            d12 = slim.conv2d(d11, 64, [3,3])
            p1 = tf.layers.max_pooling2d(d12, [2,2], 2, padding='same')

            d21 = slim.conv2d(p1, 128, [3,3])
            d22 = slim.conv2d(d21, 128, [3,3])
            p2 = tf.layers.max_pooling2d(d22, [2,2], 2, padding='same')
            
            d31 = slim.conv2d(p2, 256, [3,3])
            d32 = slim.conv2d(d31, 256, [3,3])
            p3 = tf.layers.max_pooling2d(d32, [2,2], 2, padding='same')

            d41 = slim.conv2d(p3, 512, [3,3])
            d42 = slim.conv2d(d41, 512, [3,3])
            p4 = tf.layers.max_pooling2d(d42, [2,2], 2, padding='same')
            
            d51 = slim.conv2d(p4, 1024, [3,3])
            d52 = slim.conv2d(d51, 1024, [3,3])

            ### Decoder ####################################
            
            u41 = slim.conv2d(tf.depth_to_space(d52, 2), 512, [3,3])
            u42 = slim.conv2d(tf.concat([u41, d42], axis=-1), 512, [3,3])
            u43 = slim.conv2d(u42, 512, [3,3])
            
            u31 = slim.conv2d(tf.depth_to_space(u43, 2), 128, [3,3])
            u32 = slim.conv2d(tf.concat([u31, d32], axis=-1), 128, [3,3])
            u33 = slim.conv2d(u32, 128, [3,3])

            u21 = slim.conv2d(tf.depth_to_space(u33, 2), 64, [3,3])
            u22 = slim.conv2d(tf.concat([u21, d22], axis=-1), 64, [3,3])
            u23 = slim.conv2d(u22, 64, [3,3])

            u11 = slim.conv2d(tf.depth_to_space(u23, 2), 32, [3,3])
            u12 = slim.conv2d(tf.concat([u11, d12], axis=-1), 32, [3,3])
            u13 = slim.conv2d(u12, 32, [3,3])

            prob_feat = slim.conv2d(u13, 2, [1,1],
                normalizer_fn=None,
                activation_fn=None,
                padding='SAME')
            
            pred = tf.nn.softmax(prob_feat, name='pred')
            mask = tf.argmax(pred, axis=-1, name='mask')
            
            return prob_feat, mask

def model_fn(features, labels, mode, hparams):
   
    del labels
    
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    im_l = tf.concat([features['img'], features['label']], axis=-1)
    x = tf.image.random_flip_left_right(im_l)
    '''
    if is_training:
        x = tf.contrib.image.rotate(x, tf.random.normal([FLAGS.batch_size]))
        x = tf.image.random_crop(x, [FLAGS.batch_size, 2*vh, 2*vw, 5])
        x = utils.distort(x, tf.placeholder_with_default([-0.0247903, 0.05102395,
            -0.03482873, 0.00815826], [4]))
    '''
    x = tf.image.resize_images(x, [vh, vw])
    features['img'] = x[:,:,:,:3]
    features['label'] = tf.cast(x[:,:,:,3:], tf.bool)

    images = features['img']
    labels = features['label']
    prob_feat, mask = unet(images, is_training)
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prob_feat,
    #                labels=labels))
    seg = tf.clip_by_value(tf.nn.softmax(prob_feat), 1e-6, 1.0)

    labels = tf.cast(labels, tf.float32)
    loss = tf.reduce_mean(  
         -tf.reduce_sum(labels * tf.log(seg), axis=-1))
    
    with tf.variable_scope("stats"):
        tf.summary.scalar("loss", loss)

    eval_ops = {
              "Test Error": tf.metrics.mean(loss),
    }
    
    def touint8(img):
        return tf.cast(img * 255.0, tf.uint8)
    im = touint8(images[0])
    
    to_return = {
          "loss": loss,
          "eval_metric_ops": eval_ops,
          'pred': mask[0],
          'im': im,
          'label': tf.argmax(labels[0], axis=-1)
    }

    predictions = {
        'mask': mask,
    }
    
    to_return['predictions'] = predictions

    utils.display_trainable_parameters()

    return to_return

def _default_hparams():
    """Returns default or overridden user-specified hyperparameters."""

    hparams = tf.contrib.training.HParams(
          learning_rate=1.0e-5,
    )
    if FLAGS.hparams:
        hparams = hparams.parse(FLAGS.hparams)
    return hparams


def main(argv):
    del argv
    
    tf.logging.set_verbosity(tf.logging.ERROR)

    hparams = _default_hparams()
    
    if FLAGS.mode == 'train':
        utils.train_and_eval(
            model_dir=FLAGS.model_dir,
            model_fn=model_fn,
            input_fn=create_input_fn,
            hparams=hparams,
            steps=FLAGS.steps,
            batch_size=FLAGS.batch_size,
       )
    elif FLAGS.mode == 'predict':
        import cv2
        from matplotlib import pyplot as plt
        from gen_tfrecords import central_crop
        with tf.Session() as sess:
            unet = utils.UNet(FLAGS.model_dir, sess)
            im = central_crop(cv2.imread(FLAGS.image), vw, vh) / 255.0
            t = time()
            mask = unet.run(im)
            print("Inference took %f ms" % (1000*(time()-t)))
            image = .3 * im + .7 * np.squeeze(mask)[...,np.newaxis]
            plt.imshow(image)
            plt.show()
    else:
        raise ValueError("Unknown mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    sys.excepthook = utils.colored_hook(
        os.path.dirname(os.path.realpath(__file__)))
    tf.app.run()
