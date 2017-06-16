# Copyright 2017 Paul Balanca. All Rights Reserved.
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
"""Contains model definitions for MobileNets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models import model
from models_slim import custom_layers, btree_layers

slim = tf.contrib.slim


# =========================================================================== #
# MobileNets classes.
# =========================================================================== #
class MobileNetsBTreeModel(model.Model):
    def __init__(self, model='mobilenets',
                 kernel_size=[3, 3], width_multiplier=1.0, dropouts=[0.5]):
        super(MobileNetsBTreeModel, self).__init__(model, 224, 64, 0.005)
        self.width_multiplier = width_multiplier
        self.kernel_size = kernel_size
        self.dropouts = dropouts

    def inference(self, images, num_classes,
                  is_training=True, data_format='NCHW', data_type=tf.float32):
        # Define VGG using functional slim definition
        arg_scope = mobilenets_arg_scope(is_training=is_training, data_format=data_format)
        with slim.arg_scope(arg_scope):
            return mobilenets_btree(images, num_classes,
                                    self.kernel_size,
                                    self.width_multiplier,
                                    self.dropouts,
                                    is_training=is_training)

    def pre_rescaling(self, images, is_training=True):
        return mobilenets_pre_rescaling(images, is_training)


# =========================================================================== #
# Functional definition.
# =========================================================================== #
def mobilenets_pre_rescaling(images, is_training=True):
    """Rescales an images Tensor before feeding the network
    Input tensor supposed to be in [0, 256) range.
    """
    # Rescale to [-1,1] instead of [0, 1)
    # images *= 1. / 255.
    # images = tf.subtract(images, 0.5)
    # images = tf.multiply(images, 2.0)
    # images *= 1. / 255.
    images -= 127.5
    images *= 1. / 127.5
    return images


def mobilenets_arg_scope(weight_decay=0.00004,
                         data_format='NCHW',
                         batch_norm_decay=0.9997,
                         batch_norm_epsilon=0.00001,
                         is_training=True):
    """Defines the default arg scope for MobileNets models.
    """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': True,
        'data_format': data_format,
        'is_training': is_training,
    }
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d,
                         slim.fully_connected,
                         custom_layers.depthwise_convolution2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d, slim.separable_conv2d,
                 custom_layers.depthwise_convolution2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                padding='SAME'):
            # Data format scope...
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d,
                                 slim.max_pool2d, slim.avg_pool2d,
                                 custom_layers.pad2d,
                                 custom_layers.depthwise_convolution2d,
                                 custom_layers.concat_channels,
                                 custom_layers.channel_to_last,
                                 custom_layers.spatial_squeeze,
                                 custom_layers.spatial_mean,
                                 btree_layers.conv2d_1x1_split,
                                 btree_layers.translate_channels],
                                data_format=data_format) as sc:
                return sc


def mobilenets_btree(inputs,
                     num_classes=1000,
                     kernel_size=[3, 3],
                     width_multiplier=1.0,
                     dropouts=[0.5],
                     pad_logits=True,
                     is_training=True,
                     reuse=None,
                     scope='MobileNets'):
    """MobileNets implementation.
    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
        scope: Optional scope for the variables.

    Returns:
        the last op containing the log predictions and end_points dict.
    """
    # MobileNets kernel size and padding (for layers with stride > 1).
    # kernel_size = [3, 3]
    padding = [(kernel_size[0]-1)//2, (kernel_size[1]-1)//2]

    def mobilenet_block(net, num_out_channels, stride=[1, 1],
                        scope=None):
        """Basic MobileNet block combining:
         - depthwise conv + BN + relu
         - 1x1 conv + BN + relu
        """
        with tf.variable_scope(scope, 'block', [net]) as sc:
            num_out_channels = int(num_out_channels * width_multiplier)
            if stride[0] == 1 and stride[1] == 1:
                # Depthwise convolution with stride=1
                net = custom_layers.depthwise_convolution2d(
                    net, kernel_size,
                    depth_multiplier=1, stride=stride,
                    scope='conv_dw')
            else:
                # Mimic CAFFE padding if stride > 1 => usually better accuracy.
                net = custom_layers.pad2d(net, pad=padding)
                net = custom_layers.depthwise_convolution2d(
                    net, kernel_size, padding='VALID',
                    depth_multiplier=1, stride=stride,
                    scope='conv_dw')
            # Pointwise convolution.
            net = slim.conv2d(net, num_out_channels, [1, 1],
                              scope='conv_pw')
            return net

    def mobilenet_block_btree(net, num_out_channels, stride=[1, 1],
                              scope=None):
        """Basic MobileNet block combining:
         - depthwise conv + BN + relu
         - 1x1 conv + BN + relu
        """
        with tf.variable_scope(scope, 'block', [net]) as sc:
            num_out_channels = int(num_out_channels * width_multiplier)
            # Depthwise convolution with stride=1
            net = custom_layers.depthwise_convolution2d(
                net, kernel_size,
                depth_multiplier=1, stride=stride,
                scope='conv_dw')
            # Split-pointwise convolution.
            net = btree_layers.conv2d_1x1_split(
                net, num_out_channels, split=2, scope='conv_pw_split')
            return net

    with tf.variable_scope(scope, 'MobileNets', [inputs], reuse=reuse) as sc:
        end_points = {}
        # First full convolution...
        net = custom_layers.pad2d(inputs, pad=padding)
        net = slim.conv2d(net, 32, kernel_size, stride=[2, 2],
                          padding='VALID', scope='conv1')
        # net = slim.conv2d(inputs, 32, kernel_size, stride=[2, 2],
        #                   padding='SAME', scope='conv1')
        # Then, MobileNet blocks!
        net = mobilenet_block(net, 64, scope='block2')
        net = mobilenet_block(net, 128, stride=[2, 2], scope='block3')
        net = mobilenet_block(net, 128, scope='block4')
        net = mobilenet_block(net, 256, stride=[2, 2], scope='block5')
        net = mobilenet_block(net, 256, scope='block6')
        net = mobilenet_block(net, 512, stride=[2, 2], scope='block7')
        # Intermediate blocks...
        for i in range(5):
            # Residual block...
            res = net
            net = mobilenet_block_btree(net, 512, scope='block%i_a' % (i+8))
            net = btree_layers.translate_channels(
                net, delta=128, scope='ch_translate_%i' % (i+8))
            net = mobilenet_block_btree(net, 512, scope='block%i_b' % (i+8))
            net = res + net

        # Final blocks.
        net = mobilenet_block(net, 1024, stride=[2, 2], scope='block13')
        net = mobilenet_block(net, 1024, scope='block14')
        # Spatial pooling + fully connected layer.
        net = custom_layers.spatial_mean(net, keep_dims=True, scope='spatial_mean14')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          normalizer_params=None,
                          biases_initializer=tf.zeros_initializer(),
                          scope='conv_fc15')
        net = custom_layers.spatial_squeeze(net)

        # Logits padding: get everyone to the same number of classes.
        if pad_logits:
            net = custom_layers.pad_logits(net, pad=(num_classes - 1000, 0))
        return net, end_points
mobilenets_btree.default_image_size = 224

