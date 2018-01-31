
"""分類イメージ/ラベルのペアを返すファクトリパターンクラス."""

from __future__ import division
from __future__ import print_function

import flowers


datasets_map = {
    'flowers': flowers,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
  """データセット名が与えられ、split_nameがデータセットを返します。

   Args：
     name：String、データセットの名前。
     split_name：電車/テストのスプリット名。
     dataset_dir：データセットファイルが格納されているディレクトリ。
     file_pattern：データセットのソースファイルの照合に使用するファイルパターン。
     reader：tf.ReaderBaseのサブクラスです。 `None`のままにすると、デフォルト
       各データセットで定義されたリーダーが使用されます。

   戻り値：
     `Dataset`クラスです。

   発生する：
     ValueError：データセット `name`が不明な場合。
  """
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  return datasets_map[name].get_split(
      split_name,
      dataset_dir,
      file_pattern,
      reader)
