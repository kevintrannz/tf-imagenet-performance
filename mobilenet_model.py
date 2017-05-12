# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""MobileNet model configuration.
"""

from six.moves import xrange  # pylint: disable=redefined-builtin

import model as model_lib


def mobilenet_block(cnn, num_out_channels, d_height=1, d_width=1):
    """Basic MobileNet block: depthwise conv + 1x1 conv.
    """
    cnn.conv_dw(3, 3, d_height, d_width)
    cnn.conv(num_out_channels, 1, 1)


class MobileNetModel(model_lib.Model):
    """Resnet V1 cnn network configuration."""

    def __init__(self, model, layer_counts=[5]):
        defaults = {
            'mobilenet': 32,
        }
        batch_size = defaults.get(model, 32)
        super(MobileNetModel, self).__init__(model,
                                             image_size=224,
                                             batch_size=batch_size,
                                             learning_rate=0.005,
                                             layer_counts=layer_counts)

    def add_inference(self, cnn):
        if self.layer_counts is None:
            raise ValueError('Layer counts not specified for %s' % self.get_model())
        # Batch norm parameters.
        cnn.use_batch_norm = True
        cnn.batch_norm_config = {'decay': 0.997, 'epsilon': 1e-5, 'scale': True}

        # First full convolution...
        cnn.conv(32, 3, 3, 2, 2)
        # Then, MobileNet blocks!
        mobilenet_block(cnn, 64)
        mobilenet_block(cnn, 128, 2, 2)
        mobilenet_block(cnn, 128)
        mobilenet_block(cnn, 256, 2, 2)
        mobilenet_block(cnn, 256)
        mobilenet_block(cnn, 512, 2, 2)
        # Intermediate blocks...
        for i in xrange(self.layer_counts[0]):
            mobilenet_block(cnn, 512)
        # Final ones + spatial pooling.
        mobilenet_block(cnn, 1024, 2, 2)
        mobilenet_block(cnn, 1024)
        cnn.spatial_mean()

