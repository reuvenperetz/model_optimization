# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
import numpy as np
from keras.layers import Conv2D

from model_compression_toolkit.core.keras.constants import ACTIVATION, LINEAR
from tests.keras_tests.tpc_keras import get_quantization_disabled_keras_tpc
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import model_compression_toolkit as mct

keras = tf.keras
layers = keras.layers

class MyLayer(Conv2D):
    def __init__(self,**kwargs):
        super(MyLayer, self).__init__(**kwargs)
        # w_init = tf.random_normal_initializer()
        # self.w = tf.Variable(
        #     initial_value=w_init(shape=(input_dim, units), dtype="int8"),
        #     trainable=False,
        # )
        self.kernel2 = tf.Variable(np.random.randint(0,100,size=(3,3,3,3), dtype=np.int8),
                                     trainable=False,
                                     dtype=tf.int8)

    def call(self, inputs):
        return super(MyLayer, self).call(inputs)

    def get_config(self):
        cfg = super(MyLayer, self).get_config()
        # cfg.update({})
        return cfg


class CustomLayerTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    # def get_tpc(self):
    #     return get_quantization_disabled_keras_tpc("activation_decomp_test")

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        outputs = MyLayer(filters=3, kernel_size=3, dtype="int8")(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(isinstance(quantized_model.layers[1], layers.Conv2D))
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2], layers.Activation))
        self.unit_test.assertTrue(
            quantized_model.layers[1].get_config().get(ACTIVATION) == LINEAR)
        self.unit_test.assertTrue(
            quantized_model.layers[2].get_config().get(ACTIVATION) == self.activation_function)

