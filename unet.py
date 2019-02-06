#!/usr/bin/env python3

import os
import sys
import datetime
import tensorflow as tf
from tensorflow.contrib import slim

import numpy as np

from time import time

import utils
import test_net
import layers
from dataset.coco_classes import calc_classes, calc_class_names
N_CLASSES = len(calc_classes.keys())

from dataset.gen_tfrecords import vw as __vw
from dataset.gen_tfrecords import vh as __vh

vw = 256
vh = 192 # Need 128 since we go down by factors of 2

FLAGS = tf.app.flags.FLAGS
if __name__ == '__main__':
    tf.app.flags.DEFINE_string("mode", "train", "train or pr")

    tf.app.flags.DEFINE_string("model_dir", "model", "Estimator model_dir")
    tf.app.flags.DEFINE_string("data_dir", "dataset/CampusLoopDataset", "Path to data")
    tf.app.flags.DEFINE_string("title", "Precision-Recall Curve", "Plot title")
    tf.app.flags.DEFINE_integer("n_include", 1, "")

    tf.app.flags.DEFINE_integer("steps", 2000000, "Training steps")
    tf.app.flags.DEFINE_string(
        "hparams", "",
        "A comma-separated list of `name=value` hyperparameter values. This flag "
        "is used to override hyperparameter settings when manually "
        "selecting hyperparameters.")

    tf.app.flags.DEFINE_integer("batch_size", 12, "Size of mini-batch.")
    tf.app.flags.DEFINE_string("input_dir", "/mnt/f3be6b3c-80bb-492a-98bf-4d0d674a51d6/coco/calc_tfrecords/", "tfrecords dir")

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
        tfrecord = 'train_data*.tfrecord' if split=='train' else 'validation_data.tfrecord'

        def parser(serialized_example):


            features_ = {}
            features_['img'] = tf.FixedLenFeature([], tf.string)
            features_['label'] = tf.FixedLenFeature([], tf.string)

            if split!='train':
                features_['cl_live'] = tf.FixedLenFeature([], tf.string)
                features_['cl_mem'] = tf.FixedLenFeature([], tf.string)

            fs = tf.parse_single_example(
                serialized_example,
                features=features_
            )
            

            fs['img'] = tf.reshape(tf.cast(tf.decode_raw(fs['img'], tf.uint8),
                tf.float32) / 255.0, [__vh,__vw,3])
            fs['label'] = tf.reshape(tf.cast(tf.decode_raw(fs['label'], tf.uint8),
                tf.float32), [__vh,__vw,N_CLASSES])

            if split!='train':
                fs['cl_live'] = tf.reshape(tf.cast(tf.decode_raw(fs['cl_live'], tf.uint8),
                    tf.float32) / 255.0, [__vh,__vw,3])
                fs['cl_mem'] = tf.reshape(tf.cast(tf.decode_raw(fs['cl_mem'], tf.uint8),
                    tf.float32) / 255.0, [__vh,__vw,3])
                fs['cl_live'] = tf.reshape(tf.image.resize_images(fs['cl_live'],
                    (vh, vw)), [vh,vw,3])
                fs['cl_mem'] = tf.reshape(tf.image.resize_images(fs['cl_mem'],
                    (vh, vw)), [vh,vw,3])
               
            return fs

        if split=='train':
            files = tf.data.Dataset.list_files(indir + tfrecord, shuffle=True,
                    seed=np.int64(time()))
        else:
            files = [indir + tfrecord]
            
        dataset = tf.data.TFRecordDataset(files)
        if split=='train':
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(400, seed=np.int64(time())))
        dataset = dataset.apply(tf.data.experimental.map_and_batch(parser,
            batch_size if split=='train' else batch_size//3,
            num_parallel_calls=2))
        dataset = dataset.prefetch(buffer_size=2)

        return dataset

    return input_fn


def vss(images, is_training=False, ret_descr=False):
    
    bn = {
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training,
        'fused': True,  # Use fused batch norm if possible.
    }

    # Variational Semantic Segmentator
    with tf.variable_scope("VSS"):
        im_flip = tf.image.flip_left_right(images)
        inpt = tf.identity(tf.concat([images, im_flip],
            axis=-1), name='inpt')

        center =  tf.placeholder_with_default(np.array([[0,0]],dtype=np.float32),
                shape=[1,2], name='center')


        c0 = slim.conv2d(inpt, 256, [3,3], scope="c0",
             padding='SAME',
             normalizer_fn=slim.batch_norm,
             normalizer_params=bn,
             weights_initializer=tf.contrib.layers.xavier_initializer(False),
             activation_fn=lambda x: tf.nn.elu(x)
        )
            
        b0 = layers.res_block(c0, bn, name='b0')

        c1d = layers.learned_downsample(b0, bn, name='c1d')       
        #cb1 = layers.res_block(c1d, bn, name='cb1')

        c2d = layers.learned_downsample(c1d, bn, name='c2d')
        #cb2 = layers.res_block(c2d, bn, name='cb2')
        
        p = layers.polar_transformer(c2d, center, tf.shape(c2d)[1:3])

        c3d = layers.learned_downsample(p, bn, name='c3d')
        #cb3 = layers.res_block(c3d, bn, name='cb3')
 
        c4d = layers.learned_downsample(c3d, bn, name='c4d')

        cb4 = layers.res_block(c4d, bn, name='cb4')
        cf = layers.res_block(cb4, bn, name='cf')
      
        # Dont slice since we dont want to compute twice as many feature maps for nothing
        mu = slim.conv2d(cf, N_CLASSES + 3, [1,1], scope="mu",
             padding='SAME',
             normalizer_fn=None,
             weights_initializer=tf.contrib.layers.xavier_initializer(False),
             activation_fn=None
        ) 

        log_sig_sq = tf.layers.conv2d(cf, N_CLASSES + 3, [1,1], name="sigma_sq",
             padding='SAME',
             kernel_initializer=tf.contrib.layers.xavier_initializer(False),
             bias_initializer=tf.ones_initializer(),
             activation=None
        )

        ############### Decoder ###########################
        # z = mu + sigma * epsilon
        # epsilon is a sample from a N(0, 1) distribution
        eps = tf.random_normal(tf.shape(mu), 0.0, 1.0, dtype=tf.float32)

        # Random normal variable for decoder :D
        z = mu + tf.sqrt(tf.exp(log_sig_sq)) * eps

        d1u = layers.learned_upsample(z, bn, name='d1u')

        d2u = layers.learned_upsample(d1u, bn, name='d2u')
            
        sh = tf.shape(c2d)
        
        p_i = layers.polar_transformer(d2u, center,
                tf.shape(c2d)[1:3], inverse=True)
        
        d3u = layers.learned_upsample(p_i, bn, name='d3u')

        d4u = layers.learned_upsample(d3u, bn, name='d4u')

        d4b = layers.res_block(d4u, bn, name='d4b')
        df = layers.res_block(d4b, bn, name='df')

        # Still have to softmax. These are just the features!!
        rseg = slim.conv2d(df, N_CLASSES+3, [1,1], scope="rseg",
             padding='SAME',
             normalizer_fn=None,
             weights_initializer=tf.contrib.layers.xavier_initializer(False),
             activation_fn=tf.nn.sigmoid
        )

        rec = tf.identity(rseg[:,:,:,:3], name='rec')
        seg = tf.identity(rseg[:,:,:,3:], name='seg')

        if not ret_descr:
            return mu, log_sig_sq, rec, seg, z
        else:
            sh = mu.get_shape().as_list()
            return tf.math.l2_normalize(tf.reshape(tf.transpose(mu, [0, 3, 1, 2]),
                        [-1, sh[3], sh[1]*sh[2]]), axis=-1)


def model_fn(features, labels, mode, hparams):
   
    del labels
    
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    sz = FLAGS.batch_size if is_training else FLAGS.batch_size//3
    """
    R = layers.rand_rotation(tf.random.normal([sz]))
    T = tf.reshape(tf.concat([R, tf.zeros([sz,2,1])], 
        axis=-1), [sz, 6])
    im_l = tf.concat([features['img'], features['label']], axis=-1)
    im_l = layers.transformer(im_l, T, [vh, vw])
    """
    im_l = tf.concat([features['img'], features['label']], axis=-1)
    #im_l = tf.reshape(im_l, [sz, __vh, __vw, 3+len(calc_class_names)])
    im_l = tf.contrib.image.rotate(im_l, tf.random.normal([sz]))

    x = tf.image.random_flip_left_right(im_l)
    x = tf.image.random_crop(x, [sz, vh, vw, 3+len(calc_class_names)])
    im_r = x[:,:,:,:3]
    features['label'] = x[:,:,:,3:]
   
    if is_training:
        images = tf.image.random_crop(features['img'], [sz, vh, vw, 3])
        labels = features['label']
    else:
        images = tf.concat([im_r, features['cl_live'], features['cl_mem']], 0)
        labels = tf.tile(features['label'], [3, 1, 1, 1]) # Dummy data for labels[batch//3:]
        
    mu, log_sig_sq, rec, seg, z = vss(images, is_training)
    if is_training:
        segloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=seg,
                    labels=labels))
    else:
        segloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=seg[:FLAGS.batch_size//3],
                    labels=labels[:FLAGS.batch_size//3]))

    # Can still compare rec for campus loop data
    ims = im_r if is_training else images
    recloss = tf.reduce_mean(  
         -tf.reduce_sum(ims * tf.log(tf.clip_by_value(rec, 1e-10, 1.0))
         + (1.0 - ims) * tf.log(tf.clip_by_value(1.0 - rec, 1e-10, 1.0)),
         axis=[1, 2, 3]))

    sh = mu.get_shape().as_list()
    nwh = (3+N_CLASSES) * sh[1] * sh[2]
    m = tf.clip_by_value(tf.reshape(mu, [-1, nwh]), -10, 10) # [?, 16 * w*h]
    s = tf.clip_by_value(tf.reshape(log_sig_sq, [-1, nwh]), 1e-10, 10)
    # stdev is the diagonal of the covariance matrix
    # .5 (tr(sigma2) + mu^T mu - k - log det(sigma2)) approximation
    kld = tf.reduce_mean(
            -0.5 * (tf.reduce_sum(1.0 + s - tf.square(m) - tf.exp(s),
            axis=-1))) 

    kld = tf.check_numerics(kld, '\n\n\n\nkld is inf or nan!\n\n\n')
    recloss = tf.check_numerics(recloss, '\n\n\n\nrecloss is inf or nan!\n\n\n')

    loss = segloss + kld + recloss  # average over batch + class

    prob = tf.nn.softmax(seg[0])
    pred = tf.argmax(prob, axis=-1)
    mask = tf.argmax(labels[0], axis=-1)

    if not is_training:
        
        dlive = mu[(FLAGS.batch_size//3):(2*FLAGS.batch_size//3)]       
        dmem = mu[(2*FLAGS.batch_size//3):]       

        sh = mu.get_shape().as_list()
        # Compare each combination of live to mem
        tlive = tf.reshape(tf.tile(dlive,
                [tf.shape(dlive)[0], 1, 1, 1]),
                [-1, sh[1] * sh[2], sh[3]]) # [l0, l1, l2..., l0, l1, l2...]

        tmem = tf.reshape(tf.tile(tf.expand_dims(dmem, 1),
                [1, tf.shape(dlive)[0], 1, 1, 1]), 
                [-1, sh[1] * sh[2], sh[3]]) # [m0, m0, m0..., m1, m1, m1...]
        
        tlive = tf.math.l2_normalize(tf.transpose(tlive, [0,2,1]), axis=-1)
        tmem = tf.math.l2_normalize(tf.transpose(tmem, [0,2,1]), axis=-1)
        
        sim = tf.reduce_sum(tlive * tmem, axis=-1) # Cosine sim for rgb data + class data
        # Average score across rgb + classes. Map from [-1,1] -> [0,1]
        sim_agg = (1.0 + tf.reduce_mean(sim, axis=-1)) / 2.0

        sim_agg_sq = tf.reshape(sim_agg,
            [FLAGS.batch_size//3, FLAGS.batch_size//3])
        
        # Correct location is along diagonal
        labm = tf.reshape(tf.eye(FLAGS.batch_size//3,
            dtype=tf.int64), [-1])

        # ID of nearest neighbor from 
        ids = tf.argmax(sim_agg_sq, axis=-1)

        # I guess just contiguously index it?
        row_inds = tf.range(0, FLAGS.batch_size//3,
                dtype=tf.int64) * (FLAGS.batch_size//3-1)
        buffer_inds = row_inds + ids
        sim_agg_nn = tf.nn.embedding_lookup(sim_agg, buffer_inds)
        # Pull out the labels if it was correct (0 or 1)
        lab = tf.nn.embedding_lookup(labm, buffer_inds)

    def touint8(img):
        return tf.cast(img * 255.0, tf.uint8)

    _im = touint8(im_r[0])
    _rec = touint8(rec[0])



    with tf.variable_scope("stats"):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("segloss", segloss)
        tf.summary.scalar("kld", kld)
        tf.summary.scalar("recloss", recloss)
        tf.summary.histogram("z", z)
        tf.summary.histogram("mu", mu)
        tf.summary.histogram("sig", tf.exp(log_sig_sq))
        tf.summary.histogram("normal_distr", tf.random_normal(tf.shape(z)))

    eval_ops = {
              "Test Error": tf.metrics.mean(loss),
              "Seg Error": tf.metrics.mean(segloss),
              "Rec Error": tf.metrics.mean(recloss),
              "KLD Error": tf.metrics.mean(kld),
    }

    if not is_training:
        # Closer to 1 is better
        eval_ops["AUC"] = tf.metrics.auc(lab, sim_agg_nn, curve='PR')
   
    to_return = {
          "loss": loss,
          "segloss": segloss,
          "recloss": recloss,
          "kld": kld,
          "eval_metric_ops": eval_ops,
          'pred': pred,
          'rec': _rec,
          'label': mask,
          'im': _im
    }

    predictions = {
        'pred': seg,
        'rec': rec    
    }
    
    to_return['predictions'] = predictions

    utils.display_trainable_parameters()

    return to_return

def _default_hparams():
    """Returns default or overridden user-specified hyperparameters."""

    hparams = tf.contrib.training.HParams(
          learning_rate=1.0e-3,
          alpha=0.5,
          beta=0.5,
    )
    if FLAGS.hparams:
        hparams = hparams.parse(FLAGS.hparams)
    return hparams


def main(argv):
    del argv
    
    tf.logging.set_verbosity(tf.logging.ERROR)

    if FLAGS.mode == 'train':
        hparams = _default_hparams()

        utils.train_and_eval(
            model_dir=FLAGS.model_dir,
            model_fn=model_fn,
            input_fn=create_input_fn,
            hparams=hparams,
            steps=FLAGS.steps,
            batch_size=FLAGS.batch_size,
       )
    elif FLAGS.mode == 'pr':

        test_net.plot(FLAGS.model_dir, FLAGS.data_dir,
                FLAGS.n_include, FLAGS.title)
        
    else:
        raise ValueError("Unrecognized mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    sys.excepthook = utils.colored_hook(
        os.path.dirname(os.path.realpath(__file__)))
    tf.app.run()
