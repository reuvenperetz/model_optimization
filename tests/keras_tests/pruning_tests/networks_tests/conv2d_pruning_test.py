# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

import numpy as np
import tensorflow as tf

import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model

from tests.keras_tests.pruning_tests.pruning_keras_feature_test import PruningKerasFeatureTest

keras = tf.keras
layers = keras.layers


class Conv2DPruningTest(PruningKerasFeatureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test,
                         input_shape=(224, 224, 3))

    def get_tpc(self):
        tp = generate_test_tp_model({'simd_size': 1})
        return generate_keras_tpc(name="simd2_test", tp_model=tp)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=7, kernel_size=1, use_bias=False)(inputs)
        x = layers.Conv2D(filters=6, kernel_size=1, use_bias=False)(x)
        outputs = layers.Conv2D(filters=1, kernel_size=1, use_bias=False)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        from keras.applications.resnet50 import ResNet50
        model = ResNet50()
        print(sum([l.count_params() for l in model.layers]))
        return model

    def get_kpi(self):
        # sum([l.count_params() for l in model.layers]) = 55
        # Total mem = 55*4 = 220
        # cr 0.5 => 220/2 = 110
        return mct.KPI(weights_memory=25636712*3)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        quantized_model(input_x)
        float_nparams = sum([l.count_params() for l in float_model.layers])
        pruned_nparams = sum([l.count_params() for l in quantized_model.layers])
        remaining_cr = pruned_nparams/float_nparams
        print(f"Float params: {float_nparams}")
        print(f"Pruned params: {pruned_nparams}")
        print(f"Remaining CR: {remaining_cr}")

