"""Utility functions for KeypointNet.

These are helper / tensorflow related functions. The actual implementation and
algorithm is in main.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from unet import vh, vw, unet

import time
import traceback
import cv2

gtdata = None
preddata = None

N_CLASSES = 2

def display_trainable_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        #print(variable.name)
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    print("\n\nTrainable Parameters: %d\n\n" % total_parameters)

def mask_helper(im, pred, mask, title):
    h, w = pred.shape[:2]
    rgb1 = np.zeros((h, w, 3))
    rgb2 = np.zeros((h, w, 3))

    ones = np.ones((3))

    legend = []
    np.random.seed(0)
    for i in range(N_CLASSES):
        c = np.random.rand(3)
        case1 = mask==i
        case2 = pred==i
        if np.any(np.logical_or(case1, case2)):
            legend.append(Patch(facecolor=tuple(c), edgecolor=tuple(c),
                        label='background' if i==0 else 'car'))

        rgb1[case1, :] = c
        rgb2[case2, :] = c
    
   # _im = cv2.resize(im, (w,h))

    image1 = 0.3 * im + 0.7 * rgb1
    image2 = 0.3 * im + 0.7 * rgb2

    global preddata
    global gtdata

    if gtdata is None:
        plt.subplot(1,2,1)
        gtdata = plt.imshow(image1)
        f = plt.gca()
        f.axes.get_xaxis().set_ticks([])
        f.axes.get_yaxis().set_ticks([])

        plt.subplot(1,2,2)
        preddata = plt.imshow(image2)
        f = plt.gca()
        f.axes.get_xaxis().set_ticks([])
        f.axes.get_yaxis().set_ticks([])

    else:
        gtdata.set_data(image1)
        preddata.set_data(image2)

    lgd = plt.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig = plt.gcf()
    fig.suptitle(title)
    
    plt.pause(1e-9)
    plt.draw()

def log_msg(col_hdrs, row_hdr, values):
    msg = " "*(len(row_hdr)+2)
    for i in range(len(col_hdrs)):
        msg += "{0:^8s}".format(col_hdrs[i])

    msg += "\n" + " "*(len(row_hdr)+2)
    for i in range(len(col_hdrs)):
        msg += "{0:^8s}".format("-"*len(col_hdrs[i]))

    msg += "\n" + row_hdr + ": "
    for i in range(len(col_hdrs)):
        msg += "{0:^8.3f}".format(values[col_hdrs[i]])
    msg += "\n"
    print(msg)



class TrainingHook(tf.train.SessionRunHook):
    """A utility for displaying training information such as the loss, percent
    completed, estimated finish date and time."""

    def __init__(self, steps, eval_steps):
        self.steps = steps
        self.eval_steps = eval_steps
        self.last_time = time.time()
        self.last_est = self.last_time

        self.eta_interval = int(math.ceil(0.1 * self.steps))
        self.current_interval = 0

    def before_run(self, run_context):
        graph = tf.get_default_graph()
        runargs = {
            "loss": graph.get_collection("total_loss")[0],
            "im": graph.get_collection("im")[0],
            "pred": graph.get_collection("pred")[0],
            "label": graph.get_collection("label")[0],
        }

        return tf.train.SessionRunArgs(runargs)


    def after_run(self, run_context, run_values):
        step = run_context.session.run(tf.train.get_global_step())
        now = time.time()

        if self.current_interval < self.eta_interval:
            self.duration = now - self.last_est
            self.current_interval += 1
        if step % self.eta_interval == 0:
            self.duration = now - self.last_est
            self.last_est = now

        eta_time = float(self.steps - step) / self.current_interval * \
            self.duration
        m, s = divmod(eta_time, 60)
        h, m = divmod(m, 60)
        eta = "%d:%02d:%02d" % (h, m, s)

        if step % self.eval_steps == 0:

            im = run_values.results["im"] / 255.0
            pred = run_values.results["pred"] 
            mask = run_values.results["label"] 

            mask_helper(im, pred, mask, "Train")
            tp = (step,
                  self.steps,
                  time.strftime("%a %d %H:%M:%S", time.localtime(time.time() + eta_time)),
                  eta,
                  run_values.results["loss"])

            print('\n(%d/%d): ETA: %s (%s)\n Train loss = %f' % tp)        

        self.last_time = now


class PredictionHook(tf.train.SessionRunHook):

    def __init__(self):
        pass
    
    def before_run(self, run_context):
        pass

    def after_run(self, run_context, run_values):
        pass

class EvalHook(tf.train.SessionRunHook):
    """A utility for displaying training information such as the loss, percent
    completed, estimated finish date and time."""

    def __init__(self, savedir='model/plots', show=True, save=True):
        self.i = 0 
        self.show = show
        self.save = save
        self.savedir = savedir

        if not os.path.isdir(savedir):
            os.makedirs(savedir)                        

    def before_run(self, run_context):
        graph = tf.get_default_graph()
        runargs = {
            "loss": graph.get_collection("total_loss")[0],
            "im": graph.get_collection("im")[0],
            "pred": graph.get_collection("pred")[0],
            "label": graph.get_collection("label")[0],
        }
        return tf.train.SessionRunArgs(runargs)

    def after_run(self, run_context, run_values):

        step = run_context.session.run(tf.train.get_global_step())
        
        if self.i == 0:
            
            im = run_values.results["im"] / 255.0
            pred = run_values.results["pred"] 
            mask = run_values.results["label"] 

            mask_helper(im, pred, mask, "Test")
            tp = (run_values.results["loss"])

            print('Test Error = %f' % tp)        
            fl = self.savedir + "/segmentation_iteration_%d.png" % step
            plt.savefig(fl, bbox_inches='tight', dpi=100)          

        self.i += 1

def standard_model_fn(func, steps, run_config, 
        optimizer_fn=None, eval_steps=32, model_dir='model'):
    """Creates model_fn for tf.Estimator.

    Args:
    func: A model_fn with prototype model_fn(features, labels, mode, hparams).
    steps: Training steps.
    run_config: tf.estimatorRunConfig (usually passed in from TF_CONFIG).
        synchronous training.
    optimizer_fn: The type of the optimizer. Default to Adam.

    Returns:
    model_fn for tf.estimator.Estimator.
    """

    def fn(features, labels, mode, params):
        """Returns model_fn for tf.estimator.Estimator."""

        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        ret = func(features, labels, mode, params)
        tf.add_to_collection("total_loss", ret["loss"])
        tf.add_to_collection("im", ret["im"])
        tf.add_to_collection("pred", ret["pred"])
        tf.add_to_collection("label", ret["label"])
        
        train_op = None

        training_hooks = []
        
        if is_training:

            plt.ion()


            training_hooks.append(TrainingHook(steps, eval_steps))

            if optimizer_fn is None:
                optimizer = tf.train.AdamOptimizer(params.learning_rate)
            else:
                optimizer = optimizer_fn

            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5)
            train_op = slim.learning.create_train_op(ret["loss"], optimizer)


        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=ret["predictions"],
            loss=ret["loss"],
            train_op=train_op,
            eval_metric_ops=ret["eval_metric_ops"],
            training_hooks=training_hooks,
            evaluation_hooks=[EvalHook(savedir=os.path.join(model_dir,'plots'))],
        )
    return fn


def num_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

def train_and_eval(model_dir,
    steps,
    batch_size,
    model_fn,
    input_fn,
    hparams,
    log_steps=32,
    save_steps=1024,
    summary_steps=1024,
    eval_start_delay_secs=600,
    eval_throttle_secs=30):
    """Trains and evaluates our model. Supports local and distributed training.

    Args:
    model_dir: The output directory for trained parameters, checkpoints, etc.
    steps: Training steps.
    batch_size: Batch size.
    model_fn: A func with prototype model_fn(features, labels, mode, hparams).
    input_fn: A input function for the tf.estimator.Estimator.
    hparams: tf.HParams containing a set of hyperparameters.
    keep_checkpoint_every_n_hours: Number of hours between each checkpoint
        to be saved.
    save_checkpoints_secs: Save checkpoints every this many seconds.
    save_summary_steps: Save summaries every this many steps.
    eval_steps: Number of steps to evaluate model.
    eval_start_delay_secs: Start evaluating after waiting for this many seconds.
    eval_throttle_secs: Do not re-evaluate unless the last evaluation was
        started at least this many seconds ago

    Returns:
    None
    """
    n_gpus = num_gpus()
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=n_gpus)

    run_config = tf.estimator.RunConfig(
      model_dir=model_dir,
      save_checkpoints_steps=save_steps,
      save_summary_steps=summary_steps,
      train_distribute=strategy,
      keep_checkpoint_max=None)
    
    """
    if os.path.isdir(model_dir):
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=model_dir,
                vars_to_warm_start=".*encoder.*")
    else:
        ws = None
    """
    estimator = tf.estimator.Estimator(
      model_dir=model_dir,
      model_fn=standard_model_fn(
          model_fn,
          steps,
          run_config,
          eval_steps=log_steps,
          model_dir=model_dir),
      params=hparams, config=run_config)
    
    train_spec = tf.estimator.TrainSpec(
      input_fn=input_fn(split="train", batch_size=batch_size),
      max_steps=steps)

    eval_spec = tf.estimator.EvalSpec(
      input_fn=input_fn(split="validation", batch_size=batch_size),
      steps=32,
      start_delay_secs=eval_start_delay_secs,
      throttle_secs=eval_throttle_secs)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def colored_hook(home_dir):
    """Colorizes python's error message.

    Args:
    home_dir: directory where code resides (to highlight your own files).
    Returns:
    The traceback hook.
    """

    def hook(type_, value, tb):
        def colorize(text, color, own=0):
            """Returns colorized text."""
            endcolor = "\x1b[0m"
            codes = {
              "green": "\x1b[0;32m",
              "green_own": "\x1b[1;32;40m",
              "red": "\x1b[0;31m",
              "red_own": "\x1b[1;31m",
              "yellow": "\x1b[0;33m",
              "yellow_own": "\x1b[1;33m",
              "black": "\x1b[0;90m",
              "black_own": "\x1b[1;90m",
              "cyan": "\033[1;36m",
            }
            return codes[color + ("_own" if own else "")] + text + endcolor

        for filename, line_num, func, text in traceback.extract_tb(tb):
            basename = os.path.basename(filename)
            own = (home_dir in filename) or ("/" not in filename)

            print(colorize("\"" + basename + '"', "green", own) + " in " + func)
            print("%s:  %s" % (
              colorize("%5d" % line_num, "red", own),
              colorize(text, "yellow", own)))
            print("  %s" % colorize(filename, "black", own))

        print(colorize("%s: %s" % (type_.__name__, value), "cyan"))
    return hook
