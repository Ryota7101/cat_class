# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""スリムモデルを複数のクローンとレプリカに展開する

＃TODO（sguada）docstring paragraph（a）ファイルの必要性を促し、
＃（b）クローンを定義する。

＃TODO（sguada）は、モデル展開の上位レベルのコンポーネントを記述します。
＃各モデルの展開は、DeploymentConfig、
＃A、B、Cをキャプチャする＃、データを読み込むinput_fnなど

複数のGPU上または複数のマシン間で簡単にモデルをトレーニングするには、
モジュールはヘルパ関数のセットを提供します： `create_clones`、
`optimize_clones`と` deploy`です。

使用法：

  g = tf.Graph（）

  ＃DeploymentConfigを設定する
  config = model_deploy.DeploymentConfig（num_clones = 2、clone_on_cpu = True）

  ＃変数を格納しているデバイスにグローバルなステップを作成します。
  tf.device（config.variables_device（））を使って：
    global_step = slim.create_global_step（）

  ＃入力を定義する
  tf.device（config.inputs_device（））を使用して：
    画像、ラベル= LoadData（...）
    inputs_queue = slim.data.prefetch_queue（（画像、ラベル））

  ＃オプティマイザを定義します。
  tf.device（config.optimizer_device（））を使用して：
    optimizer = tf.train.MomentumOptimizer（FLAGS.learning_rate、FLAGS.momentum）

  ＃損失を含むモデルを定義します。
  def model_fn（inputs_queue）：
    画像、ラベル= inputs_queue.dequeue（）
    予測= CreateNetwork（画像）
    slim.losses.log_loss（予測、ラベル）

  model_dp = model_deploy.deploy（config、model_fn、[inputs_queue]、
                                 オプティマイザ=オプティマイザ）

  ＃トレーニングを実行します。
  slim.learning.train（model_dp.train_op、my_log_dir、
                      summary_op = model_dp.summary_op）

Cloneのnamedtupleには、各呼び出しに関連付けられた値がまとめられています
model_fn：
  * outputs： `model_fn（）`への呼び出しの戻り値。
  * scope：クローンの作成に使用されたスコープ。
  *デバイス：クローンの作成に使用されたデバイス。

DeployedModel namedtupleは、複数のトレーニングに必要な値
クローン：
  * train_op：オプティマイザのトレーニングopを実行し、
    `model_fn`によって作成されたすべての更新操作。オプティマイザ
    指定されました。
  * summary_op： `model_fn`によって作成された要約を実行する操作
    およびprocess_gradients。
  * total_loss：「テンソル」は、
    `model_fn`に正則化損失を加えたものです。
  * clones： `create_clones（）`によって返される `Clone`タプルのリストです。

DeploymentConfigパラメータ：
  * num_clones：各レプリカにデプロイするモデル・クローンの数。
  * clone_on_cpu：クローンをCPU上に配置する必要がある場合はtrue。
  * replica_id：整数。モデルが存在するレプリカのインデックス
      配備された。チーフレプリカは通常0です。
  * num_replicas：使用するレプリカの数。
  * num_ps_tasks： `ps`ジョブのタスク数。レプリカを使用しない場合は0にします。
  * worker_job_name：ワーカー・ジョブの名前。
  * ps_job_name：パラメーター・サーバー・ジョブの名前。

TODO（sguada）：
   - グラフに副作用を記述する。
   - 要約とupdate_opsには何が起こりますか？
   - グラフのコレクションが変更されています。
   - これを使用する方法についてのチュートリアルを書く。
   - 複数回の呼び出しを呼び出す可能性を分析する。

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

slim = tf.contrib.slim


__all__ = ['create_clones',
           'deploy',
           'optimize_clones',
           'DeployedModel',
           'DeploymentConfig',
           'Clone',
          ]


# Namedtupleは、デプロイ時にクローンを表すために使用されます。
Clone = collections.namedtuple('Clone',
                               ['outputs',  # model_fn（）が返すものは何でも。
                                'scope',  # The scope used to create it.
                                'device',  # The device used to create.
                               ])

# DeployedModelを表すために使用されるNamedtuple。deploy（）によって返されます。
DeployedModel = collections.namedtuple('DeployedModel',
                                       ['train_op',  # The `train_op`
                                        'summary_op',  # The `summary_op`
                                        'total_loss',  # The loss `Tensor`
                                        'clones',  # A list of `Clones` tuples.
                                       ])

# DeploymentConfigのデフォルトパラメータ
_deployment_params = {'num_clones': 1,
                      'clone_on_cpu': False,
                      'replica_id': 0,
                      'num_replicas': 1,
                      'num_ps_tasks': 0,
                      'worker_job_name': 'worker',
                      'ps_job_name': 'ps'}


def create_clones(config, model_fn, args=None, kwargs=None):
  """`model_fn`を使用して、設定に従って複数のクローンを作成します。

  `model_fn（* args、** kwargs）`の戻り値は、
  名前付きタプルで作成するために使用されたスコープとデバイス
  `クローン（出力、範囲、デバイス）`

  注： `model_fn`によって生成された損失は、
  tf.GraphKeys.LOSSESコレクション。

  クローンを使用して作成した損失、要約、またはupdate_opsを回復するには：
  `` `python
    losses = tf.get_collection（tf.GraphKeys.LOSSES、clone.scope）
    集計= tf.get_collection（tf.GraphKeys.SUMMARIES、clone.scope）
    update_ops = tf.get_collection（tf.GraphKeys.UPDATE_OPS、clone.scope）
  `` ``

  展開オプションは、configオブジェクトとサポートによって指定されます
  1つまたは複数のクローンを異なるGPUと1つまたは複数のレプリカに展開する
  そのようなクローンの

  引数 `model_fn`は` config.num_clones`と呼ばれ、
  モデルクローンは `model_fn（* args、** kwargs）`となります。

  `config`が複数のレプリカでデプロイメントを指定する場合、デフォルト
  tensorflowデバイスは、 `model_fn`への呼び出しごとに適切に設定され、
  スリムな変数作成関数：モデルとグローバル変数が作成されます
  psデバイス上では、クローン操作は `worker`デバイス上にあります。

  Args：
    config：DeploymentConfigオブジェクト。
    model_fn：呼び出し可能です。 `model_fn（* args、** kwargs）`と呼ばれます。
    args： `model_fn`に渡す引数のオプションリスト。
    kwargs： `model_fn`に渡すキーワード引数のオプションリスト。

  戻り値：
    名前付きタプル `Clone`のリスト。
  """
  clones = []
  args = args or []
  kwargs = kwargs or {}
  with slim.arg_scope([slim.model_variable, slim.variable],
                      device=config.variables_device()):
    # Create clones.
    for i in range(0, config.num_clones):
      with tf.name_scope(config.clone_scope(i)) as clone_scope:
        clone_device = config.clone_device(i)
        with tf.device(clone_device):
          with tf.variable_scope(tf.get_variable_scope(),
                                 reuse=True if i > 0 else None):
            outputs = model_fn(*args, **kwargs)
          clones.append(Clone(outputs, clone_scope, clone_device))
  return clones


def _gather_clone_loss(clone, num_clones, regularization_losses):
  """単一のクローンの損失を集めます。

   Args：
     クローン：クローン名タプル。
     num_clones：デプロイされているクローンの数。
     regularization_losses：regularization_lossesの空のリスト
       クローンロスを増やす。

   戻り値：
     クローンの総損失のテンソル。 Noneでもかまいません。
  """
  # The return value.
  sum_loss = None
  # 要約が必要な損失の個々のコンポーネント。
  clone_loss = None
  regularization_loss = None
  # クローンデバイスの損失を計算して集計します。
  with tf.device(clone.device):
    all_losses = []
    clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
    if clone_losses:
      clone_loss = tf.add_n(clone_losses, name='clone_loss')
      if num_clones > 1:
        clone_loss = tf.div(clone_loss, 1.0 * num_clones,
                            name='scaled_clone_loss')
      all_losses.append(clone_loss)
    if regularization_losses:
      regularization_loss = tf.add_n(regularization_losses,
                                     name='regularization_loss')
      all_losses.append(regularization_loss)
    if all_losses:
      sum_loss = tf.add_n(all_losses)
  # クローンデバイスブロックから要約を追加します。
  if clone_loss is not None:
    tf.summary.scalar(clone.scope + '/clone_loss', clone_loss)
  if regularization_loss is not None:
    tf.summary.scalar('regularization_loss', regularization_loss)
  return sum_loss


def _optimize_clone(optimizer, clone, num_clones, regularization_losses,
                    **kwargs):
  """1つのクローンの損失と勾配を計算します。

   Args：
     オプティマイザ：tf.Optimizerオブジェクト。
     クローン：クローン名タプル。
     num_clones：デプロイされているクローンの数。
     regularization_losses：regularization_lossesの空のリスト
       クローンロスを増やす。
     ** kwargs：compute_gradients（）に渡すkwargの辞書。

   戻り値：
     タプル（clone_loss、clone_grads_and_vars）。
       - clone_loss：クローンの総損失のテンソル。 Noneでもかまいません。
       - clone_grads_and_vars：クローンの（勾配、変数）のリスト。
         空でもかまいません。
  """
  sum_loss = _gather_clone_loss(clone, num_clones, regularization_losses)
  clone_grad = None
  if sum_loss is not None:
    with tf.device(clone.device):
      clone_grad = optimizer.compute_gradients(sum_loss, **kwargs)
  return sum_loss, clone_grad


def optimize_clones(clones, optimizer,
                    regularization_losses=None,
                    **kwargs):
  """クローンの所与のリストのクローン損失および勾配を計算する。

   注：regularization_lossesは最初のクローン損失に追加されます。

   Args：
    clones： `create_clones（）`によって作成された `Clone`のリストです。
    optimizer： `Optimizer`オブジェクトです。
    regularization_losses：オプションの正規化損失のリスト。 Noneならば
      tf.GraphKeys.REGULARIZATION_LOSSESからそれらを集めます。 `[]`を
      それらを除外する。
    ** kwargs： `compute_gradients`に渡すキーワード引数のオプションリスト。

   戻り値：
    タプル（total_loss、grads_and_vars）。
      - total_loss：クローン損失の平均を含むTensor
        正則化の損失。
      - grads_and_vars：合計を含むタプル（勾配、変数）のリスト
        各変数の勾配の

  """
  grads_and_vars = []
  clones_losses = []
  num_clones = len(clones)
  if regularization_losses is None:
    regularization_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
  for clone in clones:
    with tf.name_scope(clone.scope):
      clone_loss, clone_grad = _optimize_clone(
          optimizer, clone, num_clones, regularization_losses, **kwargs)
      if clone_loss is not None:
        clones_losses.append(clone_loss)
        grads_and_vars.append(clone_grad)
      # Only use regularization_losses for the first clone
      regularization_losses = None
  # Compute the total_loss summing all the clones_losses.
  total_loss = tf.add_n(clones_losses, name='total_loss')
  # Sum the gradients across clones.
  grads_and_vars = _sum_clones_gradients(grads_and_vars)
  return total_loss, grads_and_vars


def deploy(config,
           model_fn,
           args=None,
           kwargs=None,
           optimizer=None,
           summarize_gradients=False):
  """スリムなモデルを複数のクローンに展開します。

  展開オプションは、configオブジェクトとサポートによって指定されます
  1つまたは複数のクローンを異なるGPUと1つまたは複数のレプリカに展開する
  そのようなクローンの

  引数 `model_fn`は` config.num_clones`と呼ばれ、
  モデルクローンは `model_fn（* args、** kwargs）`となります。

  オプションの引数 `optimizer`は` Optimizer`オブジェクトです。 `None`でなければ、
  配置されたモデルは、そのオプティマイザでトレーニングするように構成されています。

  `config`が複数のレプリカでデプロイメントを指定する場合、デフォルト
  tensorflowデバイスは、 `model_fn`への呼び出しごとに適切に設定され、
  スリムな変数作成関数：モデルとグローバル変数が作成されます
  psデバイス上では、クローン操作は `worker`デバイス上にあります。

  Args：
    config： `DeploymentConfig`オブジェクトです。
    model_fn：呼び出し可能です。 `model_fn（* args、** kwargs）`と呼ばれます。
    args： `model_fn`に渡す引数のオプションリスト。
    kwargs： `model_fn`に渡すキーワード引数のオプションリスト。
    optimizer：オプションの `Optimizer`オブジェクトです。渡された場合、モデルがデプロイされます
      そのオプティマイザを使用したトレーニング用です。
    summarize_gradients：グラディエントにサマリーを追加するかどうか。

  戻り値：
    `DeployedModel`名前付きタプルです。

  """
  # Gather initial summaries.
  summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

  # Create Clones.
  clones = create_clones(config, model_fn, args, kwargs)
  first_clone = clones[0]

  # 最初のクローンからupdate_opsを収集します。 これらには、
  #たとえば、model_fnによって作成されたbatch_norm変数の更新が含まれます。
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone.scope)

  train_op = None
  total_loss = None
  with tf.device(config.optimizer_device()):
    if optimizer:
      # Place the global step on the device storing the variables.
      with tf.device(config.variables_device()):
        global_step = slim.get_or_create_global_step()

      # Compute the gradients for the clones.
      total_loss, clones_gradients = optimize_clones(clones, optimizer)

      if clones_gradients:
        if summarize_gradients:
          # Add summaries to the gradients.
          summaries |= set(_add_gradients_summaries(clones_gradients))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
          train_op = tf.identity(total_loss, name='train_op')
    else:
      clones_losses = []
      regularization_losses = tf.get_collection(
          tf.GraphKeys.REGULARIZATION_LOSSES)
      for clone in clones:
        with tf.name_scope(clone.scope):
          clone_loss = _gather_clone_loss(clone, len(clones),
                                          regularization_losses)
          if clone_loss is not None:
            clones_losses.append(clone_loss)
          # Only use regularization_losses for the first clone
          regularization_losses = None
      if clones_losses:
        total_loss = tf.add_n(clones_losses, name='total_loss')

    # 最初のクローンから要約を追加します。 これらには、model_fnとoptimize_clones（）
    #または_gather_clone_loss（）のいずれかによって作成された要約が含まれています。
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone.scope))

    if total_loss is not None:
      # Add total_loss to summary.
      summaries.add(tf.summary.scalar('total_loss', total_loss))

    if summaries:
      # Merge all summaries together.
      summary_op = tf.summary.merge(list(summaries), name='summary_op')
    else:
      summary_op = None

  return DeployedModel(train_op, summary_op, total_loss, clones)


def _sum_clones_gradients(clone_grads):
  """すべてのクローンにおける各共有変数の合計勾配を計算します。

   この関数は、clone_gradsが適切にスケーリングされていることを前提としています。
   1 / num_clones。

   Args：
     clone_grads：タプルのリスト（勾配、変数）、リストごとのリスト
     `クローン`。

   戻り値：
      グラディエントが合計された（グラディエント、変数）のタプルのリスト
      すべてのクローンにわたって
  """
  sum_grads = []
  for grad_and_vars in zip(*clone_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
    grads = []
    var = grad_and_vars[0][1]
    for g, v in grad_and_vars:
      assert v == var
      if g is not None:
        grads.append(g)
    if grads:
      if len(grads) > 1:
        sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
      else:
        sum_grad = grads[0]
      sum_grads.append((sum_grad, var))
  return sum_grads


def _add_gradients_summaries(grads_and_vars):
  """ヒストグラムサマリーをグラデーションに追加します。

   注：要約はSUMMARIESコレクションにも追加されます。

   Args：
     grads_and_vars：グラデーションから変数へのペア（タプル）のリスト。

   戻り値：
     grads_and_varsの追加された要約の_list_。
  """
  summaries = []
  for grad, var in grads_and_vars:
    if grad is not None:
      if isinstance(grad, tf.IndexedSlices):
        grad_values = grad.values
      else:
        grad_values = grad
      summaries.append(tf.summary.histogram(var.op.name + ':gradient',
                                            grad_values))
      summaries.append(tf.summary.histogram(var.op.name + ':gradient_norm',
                                            tf.global_norm([grad_values])))
    else:
      tf.logging.info('Var %s has no gradient', var.op.name)
  return summaries


class DeploymentConfig(object):
  """`deploy（）`を使ってモデルをデプロイするための設定。

   このクラスのインスタンスを `deploy（）`に渡して、正確に指定することができます
   モデルを構築する方法 あなたが1つを渡さないと、インスタンスが構築されます
   デフォルトのdeployment_hparamsから使用されます。
  """

  def __init__(self,
               num_clones=1,
               clone_on_cpu=False,
               replica_id=0,
               num_replicas=1,
               num_ps_tasks=0,
               worker_job_name='worker',
               ps_job_name='ps'):
    """DeploymentConfigを作成します。

    この設定では、複数のクローンにモデルをデプロイする方法と、
    レプリカ。モデルは各レプリカで `num_clones`回複製されます。
    `clone_on_cpu`がTrueの場合、各クローンはCPU上に置かれます。

    `num_replicas`が1の場合、モデルは単一のプロセスでデプロイされます。その中で
    `worker_device`、` num_ps_tasks`、 `ps_device`は無視されます。

    `num_replicas`が1より大きい場合、` worker_device`と `ps_device`
    `worker`ジョブと` ps`ジョブ用にTensorFlowデバイスを指定しなければなりません。
    `num_ps_tasks`は正でなければなりません。

    Args：
      num_clones：各レプリカにデプロイするモデルクローンの数。
      clone_on_cpu：真のクローンがCPU上に置かれる場合。
      replica_id：整数。モデルが存在するレプリカのインデックス
        配備された。チーフレプリカは通常0です。
      num_replicas：使用するレプリカの数。
      num_ps_tasks： `ps`ジョブのタスク数。レプリカを使用しない場合は0にします。
      worker_job_name：ワーカー・ジョブの名前。
      ps_job_name：パラメーター・サーバー・ジョブの名前。

    発生する：
      ValueError：引数が無効な場合。
    """
    if num_replicas > 1:
      if num_ps_tasks < 1:
        raise ValueError('When using replicas num_ps_tasks must be positive')
    if num_replicas > 1 or num_ps_tasks > 0:
      if not worker_job_name:
        raise ValueError('Must specify worker_job_name when using replicas')
      if not ps_job_name:
        raise ValueError('Must specify ps_job_name when using parameter server')
    if replica_id >= num_replicas:
      raise ValueError('replica_id must be less than num_replicas')
    self._num_clones = num_clones
    self._clone_on_cpu = clone_on_cpu
    self._replica_id = replica_id
    self._num_replicas = num_replicas
    self._num_ps_tasks = num_ps_tasks
    self._ps_device = '/job:' + ps_job_name if num_ps_tasks > 0 else ''
    self._worker_device = '/job:' + worker_job_name if num_ps_tasks > 0 else ''

  @property
  def num_clones(self):
    return self._num_clones

  @property
  def clone_on_cpu(self):
    return self._clone_on_cpu

  @property
  def replica_id(self):
    return self._replica_id

  @property
  def num_replicas(self):
    return self._num_replicas

  @property
  def num_ps_tasks(self):
    return self._num_ps_tasks

  @property
  def ps_device(self):
    return self._ps_device

  @property
  def worker_device(self):
    return self._worker_device

  def caching_device(self):
    """変数をキャッシュするために使用するデバイスを返します。

     レプリカを使用する場合、ワーカーCPUに変数がキャッシュされます。

     戻り値：
       変数をキャッシュする必要がない場合は、デバイスストリングまたはなし。
    """
    if self._num_ps_tasks > 0:
      return lambda op: op.device
    else:
      return None

  def clone_device(self, clone_index):
    """クローンを作成するために使用されたデバイスとクローン内のすべてのオペレーション。

     Args：
       clone_index：clone_indexを表すIntです。

     戻り値：
       `tf.device（）`に適した値。

     発生する：
       ValueError： `clone_index`がクローンの数より多いか等しいかどうか。
    """
    if clone_index >= self._num_clones:
      raise ValueError('clone_index must be less than num_clones')
    device = ''
    if self._num_ps_tasks > 0:
      device += self._worker_device
    if self._clone_on_cpu:
      device += '/device:CPU:0'
    else:
      device += '/device:GPU:%d' % clone_index
    return device

  def clone_scope(self, clone_index):
    """クローンを作成するための名前スコープ。

     Args：
       clone_index：clone_indexを表すIntです。

     戻り値：
       `tf.name_scope（）`に適したname_scopeです。

     発生する：
       ValueError： `clone_index`がクローンの数より多いか等しいかどうか。
    """
    if clone_index >= self._num_clones:
      raise ValueError('clone_index must be less than num_clones')
    scope = ''
    if self._num_clones > 1:
      scope = 'clone_%d' % clone_index
    return scope

  def optimizer_device(self):
    """オプティマイザで使用するデバイス。

     戻り値：
       `tf.device（）`に適した値。
    """
    if self._num_ps_tasks > 0 or self._num_clones > 0:
      return self._worker_device + '/device:CPU:0'
    else:
      return ''

  def inputs_device(self):
    """入力を構築するために使用するデバイス。

     戻り値：
       `tf.device（）`に適した値。
    """
    device = ''
    if self._num_ps_tasks > 0:
      device += self._worker_device
    device += '/device:CPU:0'
    return device

  def variables_device(self):
    """クローン内で作成された変数に使用するデバイスを返します。

     戻り値：
       `tf.device（）`に適した値。
    """
    device = ''
    if self._num_ps_tasks > 0:
      device += self._ps_device
    device += '/device:CPU:0'

    class _PSDeviceChooser(object):
      """PSを使用する場合、変数のスリムデバイスチューザー。"""

      def __init__(self, device, tasks):
        self._device = device
        self._tasks = tasks
        self._task = 0

      def choose(self, op):
        if op.device:
          return op.device
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op.startswith('Variable'):
          t = self._task
          self._task = (self._task + 1) % self._tasks
          d = '%s/task:%d' % (self._device, t)
          return d
        else:
          return op.device

    if not self._num_ps_tasks:
      return device
    else:
      chooser = _PSDeviceChooser(device, self._num_ps_tasks)
      return chooser.choose
