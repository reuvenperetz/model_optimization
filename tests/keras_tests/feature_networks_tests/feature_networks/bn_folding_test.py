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


from abc import ABC

from tests.keras_tests.keras_hw_models import get_keras_float_model

import tensorflow as tf

from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class BaseBatchNormalizationFolding(BaseKerasFeatureNetworkTest, ABC):

    def __init__(self, unit_test):
        super(BaseBatchNormalizationFolding, self).__init__(unit_test=unit_test)

    def get_fw_hw_model(self):
        return get_keras_float_model()

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')


class Conv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(2, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class Conv2DBNConcatnFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(2, 3, padding='same')(inputs)
        x_bn = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x_bn)
        x2 = layers.Activation('tanh')(x_bn)
        x = layers.Concatenate()([x, x_bn, x2])

        return tf.keras.models.Model(inputs=inputs, outputs=x)


class Conv2DTransposeBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2DTranspose(2, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class DepthwiseConv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(1, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class DepthwiseConv2DBNFoldingHighMultiplierTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(1, padding='same', depth_multiplier=3)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


class SeparableConv2DBNFoldingTest(BaseBatchNormalizationFolding):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.SeparableConv2D(1, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)
