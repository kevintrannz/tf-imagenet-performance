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
"""Contains model definitions for versions of the Oxford VGG network, with
classic fully connected layers.
@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models import model
from models_slim import custom_layers
from models_slim import vgg

slim = tf.contrib.slim


# =========================================================================== #
# VGG classes.
# =========================================================================== #
class Vgg11FCModel(model.Model):
    def __init__(self):
        super(Vgg11FCModel, self).__init__('vgg11_fc', 224, 64, 0.005)

    def inference(self, images, num_classes,
                  is_training=True, data_format='NCHW', data_type=tf.float32):
        # Define VGG using functional slim definition
        arg_scope = vgg_arg_scope(is_training=is_training, data_format=data_format)
        with slim.arg_scope(arg_scope):
            return vgg_a(images, num_classes, is_training=is_training)

    def pre_rescaling(self, images, is_training=True):
        return vgg.vgg_pre_rescaling(images, is_training)


class Vgg16FCModel(model.Model):
    def __init__(self):
        super(Vgg16FCModel, self).__init__('vgg11_fc', 224, 64, 0.005)

    def inference(self, images, num_classes,
                  is_training=True, data_format='NCHW', data_type=tf.float32):
        # Define VGG using functional slim definition
        arg_scope = vgg_arg_scope(is_training=is_training, data_format=data_format)
        with slim.arg_scope(arg_scope):
            return vgg_16(images, num_classes, is_training=is_training)

    def pre_rescaling(self, images, is_training=True):
        return vgg.vgg_pre_rescaling(images, is_training)


class Vgg19FCModel(model.Model):
    def __init__(self):
        super(Vgg19FCModel, self).__init__('vgg11_fc', 224, 64, 0.005)

    def inference(self, images, num_classes,
                  is_training=True, data_format='NCHW', data_type=tf.float32):
        # Define VGG using functional slim definition
        arg_scope = vgg_arg_scope(is_training=is_training, data_format=data_format)
        with slim.arg_scope(arg_scope):
            return vgg_19(images, num_classes, is_training=is_training)

    def pre_rescaling(self, images, is_training=True):
        return vgg.vgg_pre_rescaling(images, is_training)


# =========================================================================== #
# Functional definition.
# =========================================================================== #
def vgg_arg_scope(weight_decay=0.0005, data_format='NCHW', is_training=True):
    """Defines the VGG arg scope.

    Args:
        weight_decay: The l2 regularization coefficient.

    Returns:
        An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            # Data format scope...
            with slim.arg_scope([slim.conv2d, slim.max_pool2d,
                                 custom_layers.channel_to_last,
                                 custom_layers.spatial_squeeze],
                                data_format=data_format) as sc:
                return sc


def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a'):
    """Oxford Net VGG 11-Layers version A Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
                To use in classification mode, resize input to 224x224.

    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
        spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
        scope: Optional scope for the variables.

    Returns:
        the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # Fully connected layers.
            net = custom_layers.channel_to_last(net)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096,  scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout6')
            net = slim.fully_connected(net, 4096,  scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout7')
            net = slim.fully_connected(net, num_classes,  scope='fc8')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points
vgg_a.default_image_size = 224


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
    """Oxford Net VGG 16-Layers version D Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
                To use in classification mode, resize input to 224x224.

    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
        spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
        scope: Optional scope for the variables.

    Returns:
        the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # Fully connected layers.
            net = custom_layers.channel_to_last(net)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096,  scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout6')
            net = slim.fully_connected(net, 4096,  scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout7')
            net = slim.fully_connected(net, num_classes,  scope='fc8')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points
vgg_16.default_image_size = 224


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19'):
    """Oxford Net VGG 19-Layers version E Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
                To use in classification mode, resize input to 224x224.

    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
        spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
        scope: Optional scope for the variables.

    Returns:
        the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # Fully connected layers.
            net = custom_layers.channel_to_last(net)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096,  scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout6')
            net = slim.fully_connected(net, 4096,  scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout7')
            net = slim.fully_connected(net, num_classes,  scope='fc8')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points
vgg_19.default_image_size = 224

# Alias
vgg_d = vgg_16
vgg_e = vgg_19
