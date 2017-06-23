import tensorflow as tf

from .models import SegmentationModel
from .utils import volumetric_batch_norm


def activation(t):
    return tf.nn.relu(t)


def bac(t, n_chans, kernel_size, training, name, data_format):
    with tf.variable_scope(name):
        t = volumetric_batch_norm(t, training=training, data_format=data_format)
        t = activation(t)
        t = tf.layers.conv3d(
            t, n_chans, kernel_size, data_format=data_format,
            use_bias=False)
        return t


# Residual Block
def res_block(t, n_chans, kernel_size, training, name, data_format):
    s = kernel_size - 1

    with tf.variable_scope(name):
        with tf.variable_scope('transform'):
            if data_format == 'channels_first':
                t2 = t[:, :, s:-s, s:-s, s:-s]

                n_chans_dif = n_chans - tf.shape(t)[1]
                if n_chans_dif != 0:
                    t2 = tf.pad(t2, [(0, 0), (0, n_chans_dif)] + [(0, 0)] * 3)
            else:
                t2 = t[:, s:-s, s:-s, s:-s, :]

                n_chans_dif = n_chans - tf.shape(t)[4]
                if n_chans_dif != 0:
                    t2 = tf.pad(t2, [(0, 0)] * 4 + [(0, n_chans_dif)])

        t1 = t
        t1 = bac(t1, n_chans, kernel_size, training, 'block_a', data_format)
        t1 = bac(t1, n_chans, kernel_size, training, 'block_b', data_format)

        t3 = t1 + t2

    return t3


def build_path(t, blocks, kernel_size, training, name, data_format):
    with tf.variable_scope(name):
        t = bac(t, blocks[0], kernel_size, training, 'BAC_0', data_format)
        t = bac(t, blocks[0], kernel_size, training, 'BAC_1', data_format)

        for i, n_chans in enumerate(blocks[1:]):
            t = res_block(t, n_chans, kernel_size, training, f'ResBlock_{i}',
                          data_format)

        return t


def build_model(t_det, t_context, kernel_size, n_classes, training, name,
                data_format, path_blocks=[30, 40, 40, 50], n_chans_common=150):
    with tf.variable_scope(name):
        t_det = build_path(t_det, path_blocks, kernel_size, training,
                           'detailed', data_format)

        t_context = tf.layers.average_pooling3d(
            t_context, 3, 3, padding='same', data_format=data_format)

        t_context = build_path(t_context, path_blocks, kernel_size, training,
                               'conext', data_format)

        with tf.variable_scope('upconv'):
            t_context_up = volumetric_batch_norm(t_context, training=training,
                                                 data_format=data_format)
            t_context_up = activation(t_context_up)
            t_context_up = tf.layers.conv3d_transpose(
                t_context_up, path_blocks[-1], kernel_size, strides=[3, 3, 3],
                data_format=data_format, use_bias=False)

        t_comm = tf.concat([t_context_up, t_det],
                           axis=1 if data_format == 'channels_first' else 4)
        t_comm = res_block(t_comm, n_chans_common, kernel_size, training,
                           name='comm', data_format=data_format)

        t = bac(t_comm, n_classes, 1, training, 'C', data_format)
        t = volumetric_batch_norm(t, training=training, data_format=data_format)
        logits = t

        return logits


class DeepMedic(SegmentationModel):
    def __init__(self, n_chans_in, n_classes, data_format='channels_last'):
        self.x_det_ph = tf.placeholder(
            tf.float32, (None, n_chans_in, None, None, None), name='x_det')
        self.x_con_ph = tf.placeholder(
            tf.float32, (None, n_chans_in, None, None, None), name='x_con')
        self.y_ph = tf.placeholder(
            tf.int64, (None, None, None, None), name='y_true')
        self._training_ph = tf.placeholder(tf.bool, name='is_training')

        if data_format == 'channels_last':
            x_det = tf.transpose(self.x_det_ph, [0, 2, 3, 4, 1])
            x_con = tf.transpose(self.x_con_ph, [0, 2, 3, 4, 1])
        elif data_format == 'channels_first':
            x_det = self.x_det_ph
            x_con = self.x_con_ph
        else:
            raise ValueError('wrong data format')

        self.logits = build_model(x_det, x_con, 3, n_classes,
                                  self.training_ph, 'deep_medic', data_format)

        if data_format == 'channels_last':
            self.logits = tf.transpose(self.logits, [0, 4, 1, 2, 3])

        with tf.name_scope('predict_proba'):
            self.y_pred_proba = tf.nn.softmax(self.logits, 1)

        with tf.name_scope('predict'):
            self._y_pred = tf.argmax(self.logits, axis=1)

        with tf.name_scope('loss'):
            self._loss = tf.losses.sparse_softmax_cross_entropy(
                self.y_ph, tf.transpose(self.logits, [0, 2, 3, 4, 1]))

    @property
    def graph(self):
        return tf.get_default_graph()

    @property
    def x_phs(self):
        return [self.x_det_ph, self.x_con_ph]

    @property
    def y_phs(self):
        return [self.y_ph]

    @property
    def training_ph(self):
        return self._training_ph

    @property
    def loss(self):
        return self._loss

    @property
    def y_pred(self):
        return self._y_pred
