
"""TF-ExampleプロトタイプのTFRコードにダウンロードし、花のデータを変換します。

このモジュールは、花のデータをダウンロードし、解凍し、ファイルを読み込みます
Flowersデータを構成し、2つのTFRecordデータセットを作成します.1つは電車用
1つはテスト用です。 各TFRecordデータセットは、一連のTF-Example
プロトコルバッファーであり、それぞれに単一のイメージとラベルが含まれています。

スクリプトの実行には約1分かかります。

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

# The URL where the Flowers data can be downloaded.
_DATA_URL = 'http://vps7-d.kuku.lu/files/20180131-1001_8034aa41b1244d33a7524d2cf8fcc0a1.zip'

# 検証セット内のイメージの数。
_NUM_VALIDATION = 60

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 3


class ImageReader(object):
  """TensorFlowイメージコーディングユーティリティを提供するヘルパークラス."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
  """ファイル名と推論されたクラス名のリストを返します。

   Args：
     dataset_dir：一連のサブディレクトリを含むディレクトリ。
       クラス名。 各サブディレクトリには、PNGまたはJPGでエンコードされた画像が含まれている必要があります。

   戻り値：
     `dataset_dir`に関連するイメージファイルパスのリストと
     クラス名を表すサブディレクトリ
  """
  flower_root = os.path.join(dataset_dir, 'flower_photos')
  directories = []
  class_names = []
  for filename in os.listdir(flower_root):
    path = os.path.join(flower_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'flowers_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
  """与えられたファイル名をTFRecordデータセットに変換します。

   Args：
     split_name：データセットの名前。「train」または「validation」。
     filenames：pngまたはjpgイメージへの絶対パスのリスト。
     class_names_to_ids：クラス名（文字列）からIDへの辞書
       （整数）。
     dataset_dir：変換されたデータセットが格納されているディレクトリ。
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
  """データセットの作成に使用された一時ファイルを削除します。

   Args：
     dataset_dir：一時ファイルが格納されているディレクトリ。
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'flower_photos')
  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir):
  """ダウンロードおよび変換操作を実行します。

   Args：
     dataset_dir：データセットが格納されているデータセットディレクトリ。
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[_NUM_VALIDATION:]
  validation_filenames = photo_filenames[:_NUM_VALIDATION]

  # まず、トレーニングセットと検証セットを変換します。
  _convert_dataset('train', training_filenames, class_names_to_ids,
                   dataset_dir)
  _convert_dataset('validation', validation_filenames, class_names_to_ids,
                   dataset_dir)

  # 最後に、ラベルファイルを作成します。
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the Flowers dataset!')
