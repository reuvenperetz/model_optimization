# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

import tensorflow as tf
from keras.layers import Conv2D, Dense, Reshape, Flatten, Cropping2D, ZeroPadding2D, \
    Dropout, MaxPooling2D, ReLU, Activation, Add, PReLU, BatchNormalization, Conv2DTranspose, DepthwiseConv2D

from model_compression_toolkit.common.hardware_representation.hardware2framework import \
    FrameworkHardwareModel, LayerFilterParams
from model_compression_toolkit.common.hardware_representation.hardware2framework import \
    OperationsSetToLayers
from model_compression_toolkit.hardware_models.imx500 import get_imx500_model


def get_imx500_keras():
    imx500hm = get_imx500_model()
    imx500_keras = FrameworkHardwareModel(imx500hm,
                                          name='imx500_keras')

    with imx500_keras:
        OperationsSetToLayers("NoQuantization", [Reshape,
                                                 tf.reshape,
                                                 Flatten,
                                                 Cropping2D,
                                                 ZeroPadding2D,
                                                 Dropout,
                                                 MaxPooling2D,
                                                 tf.split,
                                                 tf.quantization.fake_quant_with_min_max_vars,
                                                 BatchNormalization])

        OperationsSetToLayers("Conv", [Conv2D,
                                       DepthwiseConv2D])

        OperationsSetToLayers("FullyConnected", [Dense])

        OperationsSetToLayers("ConvTranspose", [Conv2DTranspose])

        OperationsSetToLayers("AnyReLU", [tf.nn.relu,
                                          tf.nn.relu6,
                                          LayerFilterParams(ReLU, negative_slope=0.0),
                                          LayerFilterParams(Activation, activation="relu")])

        OperationsSetToLayers("Add", [tf.add,
                                      Add])

        OperationsSetToLayers("PReLU", [PReLU])

        OperationsSetToLayers("Swish", [tf.nn.swish,
                                        LayerFilterParams(Activation, activation="swish")])

        OperationsSetToLayers("Sigmoid", [tf.nn.sigmoid,
                                          LayerFilterParams(Activation, activation="sigmoid")])

        OperationsSetToLayers("Tanh", [tf.nn.tanh,
                                       LayerFilterParams(Activation, activation="tanh")])

    return imx500_keras


KERAS_IMX500_MODEL = get_imx500_keras()
