
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', '使用するTensorFlowマスターのアドレス.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'チェックポイントとイベントログが書き込まれるディレクトリ.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'デプロイするモデルクローンの数.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'CPUを使用してクローンを配備する.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'ワーカーレプリカの数.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'パラメータサーバの数。 値が0の場合、パラメータはワーカーによってローカルで処理されます。')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'データセットからデータを読み取る並列リーダーの数.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'バッチを作成するために使用されるスレッドの数.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'ログが印刷される頻度.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'サマリーが保存される頻度（秒単位）.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'モデルが保存される頻度（秒単位）.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'トレーニングを実行しているレプリカのタスクID.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'Tモデル重みの減量.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'adadeltaの減衰率.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'AdaGradアキュムレータの開始値.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    '第1瞬間推定の指数関数的減衰率.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    '第2瞬間推定の指数関数的減衰率.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'オプティマイザのイプシロン項.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          '学習率のパワー.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'FTRLアキュムレータの開始値.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    '学習率がどのように減衰するかを指定します. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    '多項式崩壊学習率によって使用される最小終了学習率.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'ラベルスムージングの量.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, '学習率減衰係数.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    '学習率が低下した後のエポックの数.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'トレーニング中にレプリカを同期させるかどうか.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'パラメータを更新する前に収集する勾配の数.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    '移動平均に使用する減衰。'
    'Noneのままにすると、移動平均は使用されません.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'ロードするデータセットの名前.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'データセットファイルが格納されているディレクトリ.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'データセット内のラベルのオフセット。 このフラグは主に、'
    'ImageNetデータセットのバックグラウンドクラスを使用しないVGGおよびResNetアーキテクチャを評価するために使用されます.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'T訓練する建築の名前.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, '使用する前処理の名前です。 `None`のままにすると、model_nameフラグが使用されます.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    '微調整するチェックポイントへのパス.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'チェックポイントからリストアするときに除外する変数のスコープのカンマ区切りリスト。')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    '列挙する変数のセットをフィルタリングするためのスコープのカンマ区切りリスト。デフォルトでは、Noneはすべての変数.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'チェックポイントを復元すると、欠落している変数は無視されます.')

FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
 
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer


def _get_init_fn():
  
  if FLAGS.checkpoint_path is None:
    return None

  # train_dirにチェックポイントが存在する場合、ユーザーに警告します。 とにかくチェックポイントを無視します。
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
  
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        weight_decay=FLAGS.weight_decay,
        is_training=True)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
      provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=FLAGS.num_readers,
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
      [image, label] = provider.get(['image', 'label'])
      label -= FLAGS.labels_offset

      train_image_size = FLAGS.train_image_size or network_fn.default_image_size

      image = image_preprocessing_fn(image, train_image_size, train_image_size)

      images, labels = tf.train.batch(
          [image, label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)
      labels = slim.one_hot_encoding(
          labels, dataset.num_classes - FLAGS.labels_offset)
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)

    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """network_fnの複数のクローンを作成してデータの並列処理を可能にする."""
      images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(images)

      #############################
      # Specify the loss function #
      #############################
      if 'AuxLogits' in end_points:
        slim.losses.softmax_cross_entropy(
            end_points['AuxLogits'], labels,
            label_smoothing=FLAGS.label_smoothing, weights=0.4,
            scope='aux_loss')
      slim.losses.softmax_cross_entropy(
          logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # 最初のクローンからupdate_opsを収集します。 これらには、
    #たとえば、network_fnによって作成されたbatch_norm変数の更新が含まれます。
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    if FLAGS.sync_replicas:
      # sync_replicasが有効な場合、平均化はチーフ・キュー・ランナーで行われます。
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          total_num_replicas=FLAGS.worker_replicas,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables)
    elif FLAGS.moving_average_decay:
      # トレーナーがローカルで実行するopsを更新します。
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # 最初のクローンから要約を追加します。 これらには、model_fnとoptimize_clones（）
    #または_gather_clone_loss（）のいずれかによって作成された要約が含まれています。
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')


    ###########################
    # Kicks off the training. #
    ###########################
    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        sync_optimizer=optimizer if FLAGS.sync_replicas else None)


if __name__ == '__main__':
  tf.app.run()
