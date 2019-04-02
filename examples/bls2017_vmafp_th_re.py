# -*- coding: utf-8 -*-
# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic nonlinear transform coder for RGB images.

This is a close approximation of the image compression model of
Ballé, Laparra, Simoncelli (2017):
End-to-end optimized image compression
https://arxiv.org/abs/1611.01704

With patches from Victor Xing <victor.t.xing@gmail.com>
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob

# Dependency imports

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc

import os

# For VMAF calculation
import errno
import os
import sys
import subprocess
import json
import logging
import random
#import threading
import threading
from Queue import Queue

def compute_vmaf(img_idx, q):
    """ given a pair of reference and distored images:
        use the ffmpeg libvmaf filter to compute vmaf, vif, ssim, and ms_ssim.
    """
    dis_image = 'tmp/rec_' + str(img_idx) + '.png'
    ref_image = 'tmp/src_' + str(img_idx) + '.png'
    width     = 128
    height    = 128
    log_path  = './tmp/stats' + str(img_idx) + '.json'
    model_path= '/work/06359/lhchen/maverick2/tools/share/model/vmaf_v0.6.1.pkl'

    libvmaf_filter_cmd = \
        'libvmaf=ssim=true:ms_ssim=false:log_fmt=json:model_path={model_path}:log_path={log_path}'.format(
           log_path=log_path,
           model_path=model_path
       )

    cmd = ['ffmpeg', '-s:v', '%s,%s' % (width, height), '-i', dis_image,
            '-s:v', '%s,%s' % (width, height), '-i', ref_image,
            '-lavfi', libvmaf_filter_cmd,
            '-f', 'null', '-'
          ]

    subprocess.check_output(" ".join(cmd), stderr=subprocess.STDOUT, shell=True)
    vmaf_log = json.load(open(log_path))
    q.put([ img_idx, vmaf_log["frames"][0]["metrics"]["vmaf"] ])


def compute_vmaf1(img_idx):
    """ given a pair of reference and distored images:
        use the ffmpeg libvmaf filter to compute vmaf, vif, ssim, and ms_ssim.
    """
    dis_image = 'tmp/rec_' + str(img_idx) + '.png'
    ref_image = 'tmp/src_' + str(img_idx) + '.png'
    width     = 128
    height    = 128
    enable_conf_interval=False
    log_path  = './tmp/stats.json'

    if enable_conf_interval:
        libvmaf_filter_cmd = \
            'libvmaf=ssim=true:ms_ssim=false:log_fmt=json:enable_conf_interval=true:model_path={model_path}:log_path={log_path}'.format(
               log_path=log_path,
               model_path='/usr/local/share/model/vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.pkl'
           )
    else:
        libvmaf_filter_cmd = \
            'libvmaf=ssim=true:ms_ssim=false:log_fmt=json:model_path={model_path}:log_path={log_path}'.format(
               log_path=log_path,
               model_path='/work/06359/lhchen/maverick2/tools/share/model/vmaf_v0.6.1.pkl'
           )

    cmd = ['ffmpeg', '-s:v', '%s,%s' % (width, height), '-i', dis_image,
            '-s:v', '%s,%s' % (width, height), '-i', ref_image,
            '-lavfi', libvmaf_filter_cmd,
            '-f', 'null', '-'
          ]

    try:
        #logger.info("\033[92m[VMAF]\033[0m " + dist_image)
        subprocess.check_output(" ".join(cmd), stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        logger.error("\033[91m[ERROR]\033[0m " + " ".join(cmd) + "\n" + e.output)

    vmaf_log = json.load(open(log_path))

    vmaf_dict = dict()
    vmaf_dict["vmaf"]    = vmaf_log["frames"][0]["metrics"]["vmaf"]
    #vmaf_dict["vif"]     = vmaf_log["frames"][0]["metrics"]["vif_scale0"]
    #vmaf_dict["adm"]     = vmaf_log["frames"][0]["metrics"]["adm2"]
    #vmaf_dict["ssim"]    = vmaf_log["frames"][0]["metrics"]["ssim"]
    #vmaf_dict["ms_ssim"] = vmaf_log["frames"][0]["metrics"]["ms_ssim"]

    if enable_conf_interval:
        # prefix _ is used to signal it is not a metric but auxiliary information
        vmaf_dict["_std_vmaf"] = vmaf_log["frames"][0]["metrics"]["stddev"]
    return vmaf_dict["vmaf"]


def load_image(filename):
  """Loads a PNG image file."""

  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def quantize_image(image):
  image = tf.clip_by_value(image, 0, 1)
  image = tf.round(image * 255)
  image = tf.cast(image, tf.uint8)
  return image


def save_image(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)


def analysis_transform(tensor, num_filters):
  """Builds the analysis transform."""

  with tf.variable_scope("analysis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

    return tensor


def synthesis_transform(tensor, num_filters):
  """Builds the synthesis transform."""

  with tf.variable_scope("synthesis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          3, (9, 9), corr=False, strides_up=4, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor


def build_model(img_ref, img_dis, b_train=False):

    out = tf.concat([img_ref, img_dis], 3)
    #out = img_ref - img_dis
    #bb_train = False
    '''
    with tf.variable_scope("conv00"):
        out = tf.layers.conv2d(out, filters=8, kernel_size=5, padding='same')
        #out = tf.layers.batch_normalization(out, training=b_train, trainable=b_train)
        #out = tf.nn.leaky_relu(out)
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 2)
    '''
    with tf.variable_scope("conv3"):
        out = tf.layers.conv2d(out, filters=16, kernel_size=5, padding='same')
        #out = tf.layers.batch_normalization(out, training=b_train, trainable=b_train)
        #out = tf.nn.leaky_relu(out)
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 2)

    with tf.variable_scope("conv1"):
        out = tf.layers.conv2d(inputs=out, filters=32, kernel_size=5, padding='same')
        #out = tf.layers.batch_normalization(out, training=b_train, trainable=b_train)
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 2)

    with tf.variable_scope("conv2"):
        out = tf.layers.conv2d(out, filters=64, kernel_size=5, padding='same')
        #out = tf.layers.batch_normalization(out, training=b_train, trainable=b_train)
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 2)

    out = tf.reshape(out, [-1, 16 * 16 * 64])
    
    with tf.variable_scope("fc"):
        pred = tf.layers.dense(out, 1)
        #pred = tf.clip_by_value(pred, 0.0, 100.0)
    return pred

def train():
  """Trains the model."""

  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device('/cpu:0'):
    train_files = glob.glob(args.train_glob)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(
        load_image, num_parallel_calls=args.preprocess_threads)
    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(32)

  num_pixels = args.batchsize * args.patchsize ** 2

  # Get training patch from dataset.
  x = train_dataset.make_one_shot_iterator().get_next()

  # Build autoencoder.
  y = analysis_transform(x, args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  y_tilde, likelihoods = entropy_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y_tilde, args.num_filters)
  x_tilde = tf.clip_by_value(x_tilde, 0.0, 1.0) #LH add

  # Total number of bits divided by number of pixels.
  train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  '''
  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  train_mse *= 255 ** 2
  '''
  '''
  train_mse = args.batchsize - tf.reduce_sum(tf.image.ssim(x, x_tilde, 1))
  train_mse = train_mse*255
  '''
  
  # Mean squared error across pixels.
  mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  mse *= 255 ** 2
  # Replacement of MSE error (by ProxIQA)
  with tf.variable_scope('model', reuse=False):
    train_mse = build_model(x, x_tilde, False)
  train_mse = tf.reduce_mean(train_mse)
  debug_prox= tf.reduce_mean(train_mse)
  #train_mse = tf.abs(100 - train_mse)
  train_mse = tf.abs(100 - train_mse)
  
  # The rate-distortion cost.
  #train_loss = args.lmbda * train_mse + train_bpp
  r = 1.0
  train_loss = args.lmbda * (r*train_mse+(2-r)*mse) + train_bpp

  # LH: define trainable variables for compression
  bls_vars = [var for var in tf.trainable_variables() if 'model' not in var.name]
  print('optimize layers for BLS')
  for var in bls_vars: print(var.name)

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step, var_list=bls_vars)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0], var_list=bls_vars)

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
  
  # LH: Saving x, x_tilde
  sv_patch_op = []
  for i in range(0,args.batchsize):
    sv_path_rec = 'tmp/rec_' + str(i) + '.png'
    sv_path_src = 'tmp/src_' + str(i) + '.png'
    sv_patch_op.append(save_image(sv_path_rec, x_tilde[i, :, :, :]))
    sv_patch_op.append(save_image(sv_path_src, x      [i, :, :, :]))
  sv_op = tf.group(*[op for op in sv_patch_op])

  # LH: SSIM (TMP)
  ssim_op = tf.image.ssim(x, x_tilde, 1)
  # LH: Debug signals
  debug_ssim = tf.reduce_mean(tf.image.ssim(x, x_tilde, 1))      #SSIM for Tensorboard
  debug_mse  = tf.reduce_mean(tf.squared_difference(x, x_tilde)) #MSE  for Tensorboard
  debug_mse *= 255 ** 2

  # LH: Training Prox IQA
  prox_vars = [var for var in tf.trainable_variables() if 'model' in var.name]
  print('optimize layers for ProxIQA')
  for var in prox_vars: print(var.name)

  x_ref   = tf.placeholder(tf.float32, shape=[None, args.patchsize, args.patchsize, 3], name='ref_img')
  x_dis   = tf.placeholder(tf.float32, shape=[None, args.patchsize, args.patchsize, 3], name='dis_img')
  y_score = tf.placeholder(tf.float32, shape=[None],                                    name='mos_score')

  with tf.variable_scope('model', reuse=True):
    prediction = build_model(x_ref, x_dis, True)
  prediction = tf.reduce_mean(prediction, axis=1)
  loss = tf.losses.mean_squared_error(labels=y_score, predictions=prediction)
  #loss = tf.losses.absolute_difference(labels=y_score, predictions=prediction)
  prox_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      train_prox_op = prox_optimizer.minimize(loss, var_list=prox_vars)

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", train_bpp)
  tf.summary.scalar("mse", train_mse)
  tf.summary.scalar("debug_mse", debug_mse)
  tf.summary.scalar("prox", debug_prox)
  tf.summary.scalar("ssim", debug_ssim)
  tf.summary.scalar("vmaf", (tf.reduce_mean(y_score)*8-98*1)/7)

  tf.summary.image("original", quantize_image(x))
  tf.summary.image("reconstruction", quantize_image(x_tilde))

  # Creates summary for the probability mass function (PMF) estimated in the
  # bottleneck.
  entropy_bottleneck.visualize()

  logged_tensors = [
        tf.identity(train_loss, name="train_loss"),
        tf.identity(train_bpp, name="train_bpp"),
        tf.identity(train_mse, name="train_mse"),
  ]
  '''
  # Reload weights from the weights subdirectory
  restore_list = ['model/conv00/conv2d/kernel:0',
                  'model/conv00/conv2d/bias:0',
                  'model/conv3/conv2d/kernel:0',
                  'model/conv3/conv2d/bias:0',
                  'model/conv1/conv2d/kernel:0',
                  'model/conv1/conv2d/bias:0',
                  'model/conv2/conv2d/kernel:0',
                  'model/conv2/conv2d/bias:0',
                  'model/fc/dense/kernel:0',
                  'model/fc/dense/bias:0']
  restore_var  = [v for v in tf.all_variables() if v.name in restore_list]
  #print(restore_var)
  #Scaffold=tf.train.Scaffold(saver=tf.train.Saver(restore_var))
  save_path = os.path.join('train_prox', 'best_weights_ssim_cat')
  if os.path.isdir(save_path):
    save_path = tf.train.latest_checkpoint(save_path)
  vmafsaver=tf.train.Saver(restore_var)
  '''

  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
      #tf.train.LoggingTensorHook(logged_tensors, every_n_secs=60),
  ]
  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=args.checkpoint_dir,
      save_checkpoint_secs=600, save_summaries_secs=60) as sess:

    #print(tf.trainable_variables())
    #vmafsaver.restore(sess, save_path) #LH add
    y_vmaf = np.zeros(args.batchsize)
    #y_vmaf_mt = np.zeros(args.batchsize)
    while not sess.should_stop():
      _, x_src, x_rec, _ = sess.run([train_op, x, x_tilde, sv_op], feed_dict={y_score: y_vmaf})
      # calculate VMAF for batches
      a_idx = range(0,args.batchsize)
      p_idx = random.sample(a_idx, random.randint(args.batchsize-1, args.batchsize)) 
      
      #for i in p_idx:
      #  y_vmaf[i]   = compute_vmaf1(i)
      
      q =Queue()
      threads = []
      for i in p_idx:
        t = threading.Thread(target=compute_vmaf,args=(i,q))
        t.start()
        threads.append(t)
      for thread in threads:
        thread.join()
      #y_vmaf_mt = []
      for _ in p_idx:
        s = q.get()
        y_vmaf[s[0]] = s[1]
      
      
      p0_idx = [i for i in a_idx if i not in p_idx]
      y_vmaf[p0_idx] = 97
      x_rec[p0_idx,:,:,:] = x_src[p0_idx,:,:,:]
      #y_vmaf_mt[p0_idx] = 97
      
      #print(y_vmaf_mt)
      #print(y_vmaf_mt - y_vmaf)

      # Training ProxIQA network
      if not sess.should_stop():
        _, proxloss, proxpred = sess.run([train_prox_op, loss, prediction], feed_dict={x_ref: x_src, x_dis: x_rec, y_score: y_vmaf})
      #print(proxloss)
      #print(proxpred)
      #print(y_vmaf)
      #sess.run(train_prox_op, feed_dict={x_ref: x_src, x_dis: x_src, y_score: np.array([1,1,1,1,1,1,1,1])})


def compress():
  """Compresses an image."""

  # Load input image and add batch dimension.
  x = load_image(args.input)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])

  # Transform and compress the image, then remove batch dimension.
  y = analysis_transform(x, args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  string = entropy_bottleneck.compress(y)
  string = tf.squeeze(string, axis=0)

  # Transform the quantized image back (if requested).
  y_hat, likelihoods = entropy_bottleneck(y, training=False)
  x_hat = synthesis_transform(y_hat, args.num_filters)

  num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))

  # Total number of bits divided by number of pixels.
  eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Bring both images back to 0..255 range.
  x *= 255
  x_hat = tf.clip_by_value(x_hat, 0, 1)
  x_hat = tf.round(x_hat * 255)

  mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
  psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
  msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    string, x_shape, y_shape = sess.run([string, tf.shape(x), tf.shape(y)])

    # Write a binary file with the shape information and the compressed string.
    with open(args.output, "wb") as f:
      f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
      f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
      f.write(string)

    # If requested, transform the quantized image back and measure performance.
    if args.verbose:
      eval_bpp, mse, psnr, msssim, num_pixels = sess.run(
          [eval_bpp, mse, psnr, msssim, num_pixels])

      # The actual bits per pixel including overhead.
      bpp = (8 + len(string)) * 8 / num_pixels

      print("Mean squared error: {:0.4f}".format(mse))
      print("PSNR (dB): {:0.2f}".format(psnr))
      print("Multiscale SSIM: {:0.4f}".format(msssim))
      print("Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim)))
      print("Information content in bpp: {:0.4f}".format(eval_bpp))
      print("Actual bits per pixel: {:0.4f}".format(bpp))


def decompress():
  """Decompresses an image."""

  # Read the shape information and compressed string from the binary file.
  with open(args.input, "rb") as f:
    x_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    y_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    string = f.read()

  y_shape = [int(s) for s in y_shape] + [args.num_filters]

  # Add a batch dimension, then decompress and transform the image back.
  strings = tf.expand_dims(string, 0)
  entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
  y_hat = entropy_bottleneck.decompress(
      strings, y_shape, channels=args.num_filters)
  x_hat = synthesis_transform(y_hat, args.num_filters)

  # Remove batch dimension, and crop away any extraneous padding on the bottom
  # or right boundaries.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

  # Write reconstructed image out as a PNG file.
  op = save_image(args.output, x_hat)

  # Load the latest model checkpoint, and perform the above actions.
  with tf.Session() as sess:
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    sess.run(op)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "command", choices=["train", "compress", "decompress"],
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options.")
  parser.add_argument(
      "input", nargs="?",
      help="Input filename.")
  parser.add_argument(
      "output", nargs="?",
      help="Output filename.")
  parser.add_argument(
      "--verbose", "-v", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=128,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="train",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--train_glob", default="images/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  parser.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  parser.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  parser.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  parser.add_argument(
      "--last_step", type=int, default=1000000,
      help="Train up to this number of steps.")
  parser.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")

  args = parser.parse_args()

  if args.command == "train":
    train()
  elif args.command == "compress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for compression.")
    compress()
  elif args.command == "decompress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for decompression.")
    decompress()
