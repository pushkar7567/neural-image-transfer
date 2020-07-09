import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import os

import numpy as np
import PIL.Image
import time
import functools

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'img')

content_path = os.path.join(UPLOAD_FOLDER, "contentfile.png")
style_path = os.path.join(UPLOAD_FOLDER, "stylefile.png")
print(content_path)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

def generate():
  content_image = load_img(content_path)
  style_image = load_img(style_path)

  plt.subplot(1, 2, 1)
  imshow(content_image, 'Content Image')

  plt.subplot(1, 2, 2)
  imshow(style_image, 'Style Image')

  import tensorflow_hub as hub
  hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
  stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
  return stylized_image
