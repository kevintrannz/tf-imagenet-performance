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
"""Some custom layers, implementing random and ludicrous ideas
I sometimes have.
"""
import math
import numpy as np
import tensorflow as tf

from models_slim import custom_layers

import functools

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.ops import variables as tf_variables

slim = tf.contrib.slim


@add_arg_scope
def split_channels(inputs, split=1, data_format='NHWC', scope=None):
    """Concat a list of tensors on the channel axis.

    Args:
      inputs: List Tensors;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'split_channels', [inputs]):
        inshape = inputs.get_shape().as_list()
        l_inputs = []
        if data_format == 'NHWC':
            # Split alongs channels...
            nchannels = inshape[-1]
            ssize = nchannels // split
            for i in range(split-1):
                l_inputs.append(inputs[:, :, :, i*ssize:(i+1)*ssize])
            l_inputs.append(inputs[:, :, :, (split-1)*ssize:])
        elif data_format == 'NCHW':
            # Split alongs channels...
            nchannels = inshape[1]
            ssize = nchannels // split
            for i in range(split-1):
                l_inputs.append(inputs[:, i*ssize:(i+1)*ssize])
            l_inputs.append(inputs[:, (split-1)*ssize:])
        return l_inputs


@add_arg_scope
def translate_channels(inputs, delta=0, data_format='NHWC', scope=None):
    """Convention:

    Positive delta: push to the right.
    Negative delta: push to the left.
    """
    with tf.name_scope(scope, 'translate_channels', [inputs]):
        if data_format == 'NHWC':
            in0 = inputs[:, :, :, :-delta]
            in1 = inputs[:, :, :, -delta:]
            outputs = custom_layers.concat_channels([in1, in0], data_format=data_format)
        else:
            in0 = inputs[:, :-delta]
            in1 = inputs[:, -delta:]
            outputs = custom_layers.concat_channels([in1, in0], data_format=data_format)
        return outputs


# =========================================================================== #
# Depthwise convolution 2d with data format option.
# =========================================================================== #
@add_arg_scope
def conv2d_1x1_split(
        inputs,
        num_outputs,
        stride=1,
        split=1,
        padding='SAME',
        activation_fn=nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        data_format='NHWC',
        scope=None):
    """1x1 convolution with splitting.
    """
    with variable_scope.variable_scope(scope, 'Conv2d_1x1_split', [inputs],
                                       reuse=reuse) as sc:
        # Split channels...
        inputs = ops.convert_to_tensor(inputs)
        l_inputs = split_channels(inputs, split=split, data_format=data_format)
        # 1x1 convolution on every component...
        nets = []
        osize = num_outputs // split
        for i in range(split):
            if i == split-1:
                osize = num_outputs - (split-1) * osize
            net = slim.conv2d(
                l_inputs[i],
                osize,
                [1, 1],
                stride=stride,
                padding=padding,
                data_format=data_format,
                rate=1,
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                weights_initializer=weights_initializer,
                weights_regularizer=weights_regularizer,
                biases_initializer=biases_initializer,
                biases_regularizer=biases_regularizer,
                reuse=reuse,
                variables_collections=variables_collections,
                outputs_collections=outputs_collections,
                trainable=trainable,
                scope='conv2d_split_%i' % i)
            nets.append(net)
        # Concat the results.
        outputs = custom_layers.concat_channels(nets, data_format=data_format)
        return outputs

# =========================================================================== #
# Extension of TensorFlow common layers.
# =========================================================================== #
