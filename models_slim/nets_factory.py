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
"""Contains a factory for building various models."""


import functools
import tensorflow as tf

from models_slim import inception
# from models_slim import resnet_v1
# from models_slim import resnet_v2
from models_slim import vgg
from models_slim import vgg_fc
from models_slim import mobilenets
from models_slim import mobilenets_pool
# from models_slim import mobilenets_caffe
from models_slim import mobilenets_leaders
from models_slim import mobilenets_btree

slim = tf.contrib.slim

networks_map = {'vgg11': vgg.vgg_a,
                'vgg16': vgg.vgg_16,
                'vgg19': vgg.vgg_19,
                'vgg11_fc': vgg_fc.vgg_a,
                'vgg16_fc': vgg_fc.vgg_16,
                'vgg19_fc': vgg_fc.vgg_19,
                'inceptionv1': inception.inception_v1,
                'inceptionv2': inception.inception_v2,
                'inceptionv3': inception.inception_v3,
                'inceptionv4': inception.inception_v4,
                'mobilenets': mobilenets.mobilenets,
                'mobilenets_k5': mobilenets.mobilenets_k5,
                'mobilenets_k7': mobilenets.mobilenets_k7,
                'mobilenets_btree': mobilenets_btree.mobilenets_btree,
                'mobilenets_caffe': mobilenets.mobilenets,
                'mobilenets_pool': mobilenets_pool.mobilenets,
                'mobilenets_leaders': mobilenets_leaders.mobilenets,
                }

arg_scopes_map = {'vgg11': vgg.vgg_arg_scope,
                  'vgg16': vgg.vgg_arg_scope,
                  'vgg19': vgg.vgg_arg_scope,
                  'vgg11_fc': vgg_fc.vgg_arg_scope,
                  'vgg16_fc': vgg_fc.vgg_arg_scope,
                  'vgg19_fc': vgg_fc.vgg_arg_scope,
                  'inceptionv1': inception.inception_v1_arg_scope,
                  'inceptionv2': inception.inception_v2_arg_scope,
                  'inceptionv3': inception.inception_v3_arg_scope,
                  'inceptionv4': inception.inception_v4_arg_scope,
                  'mobilenets': mobilenets.mobilenets_arg_scope,
                  'mobilenets_k5': mobilenets.mobilenets_arg_scope,
                  'mobilenets_k7': mobilenets.mobilenets_arg_scope,
                  'mobilenets_btree': mobilenets_btree.mobilenets_arg_scope,
                  'mobilenets_pool': mobilenets_pool.mobilenets_arg_scope,
                  'mobilenets_caffe': mobilenets.mobilenets_arg_scope,
                  'mobilenets_leaders': mobilenets_leaders.mobilenets_arg_scope,
                  }

models_map = {'vgg11': vgg.Vgg11Model(),
              'vgg16': vgg.Vgg16Model(),
              'vgg19': vgg.Vgg19Model(),
              'vgg11_fc': vgg_fc.Vgg11FCModel(),
              'vgg16_fc': vgg_fc.Vgg16FCModel(),
              'vgg19_fc': vgg_fc.Vgg19FCModel(),
              'inceptionv1': inception.Inceptionv1Model(),
              'inceptionv2': inception.Inceptionv2Model(),
              'inceptionv3': inception.Inceptionv3Model(),
              'inceptionv4': inception.Inceptionv4Model(),
              'mobilenets': mobilenets.MobileNetsModel(),
              'mobilenets_k5': mobilenets.MobileNetsModel(kernel_size=[5, 5]),
              'mobilenets_k7': mobilenets.MobileNetsModel(kernel_size=[7, 7]),
              'mobilenets_btree': mobilenets_btree.MobileNetsBTreeModel(),
              'mobilenets_caffe': mobilenets.MobileNetsCaffeModel(),
              'mobilenets_caffe_k5': mobilenets.MobileNetsCaffeModel(kernel_size=[5, 5]),
              'mobilenets_pool': mobilenets.MobileNetsModel(),
              'mobilenets_leaders': mobilenets_leaders.MobileNetsModel(),
              }


def get_network_fn(name, num_classes, is_training=False, data_format='NCHW', **kwargs):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      is_training: `True` if the model is being used for training and `False`
        otherwise.
      weight_decay: The l2 coefficient for the model weights.
    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature: logits, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    arg_scope = arg_scopes_map[name](is_training=is_training,
                                     data_format=data_format, **kwargs)
    func = networks_map[name]
    @functools.wraps(func)
    def network_fn(images, **kwargs):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training, **kwargs)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
