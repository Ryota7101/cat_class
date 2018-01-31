
"""花のデータセットのデータを提供します。

データセットの作成に使用されるデータセットスクリプトは、次の場所にあります。
テンソルフロー/モデル/研究/スリム/データセット/ download_and_convert_flowers.py
"""

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'flowers_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 240, 'validation': 60}

_NUM_CLASSES = 3

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 2',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """花を読み取るための指示を含むデータセットタプルを取得します。

   Args：
     split_name：列車/検証の分割名。
     dataset_dir：データセットソースのベースディレクトリ。
     file_pattern：データセットソースと一致するときに使用するファイルパターン。
       パターンに '％s'文字列が含まれていると仮定して、分割
       名前を挿入できます。
     reader：TensorFlowリーダータイプ。

   戻り値：
     `Dataset`名前付きタプル。

   発生する：
     ValueError： `split_name`が有効な列車/検証分割でない場合。
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # 署名でNoneを許可すると、dataset_factoryはデフォルトを使用できます。
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)