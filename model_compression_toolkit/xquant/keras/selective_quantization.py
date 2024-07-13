#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
import copy
import keras.layers


from model_compression_toolkit.xquant.common.selective_quantization import SelectiveQuantization
import tensorflow as tf
from mct_quantizers import KerasQuantizationWrapper
from mct_quantizers import KerasActivationQuantizationHolder

class KerasSelectiveQuantization(SelectiveQuantization):

    def disable_weights_quantization(self, float_model, quantized_model):

        layers_name_mapping = {}

        def _disable_weights_quantization(layer_to_substitue, float_model=float_model):
            if isinstance(layer_to_substitue, KerasQuantizationWrapper):
                original_float_layer = float_model.get_layer(layer_to_substitue.layer.name)
                result_layer = layer_to_substitue.layer
                layers_name_mapping[layer_to_substitue.name] = result_layer.name
                if hasattr(original_float_layer, 'kernel'):
                    result_layer.kernel = original_float_layer.kernel
                if hasattr(original_float_layer, 'depthwise_kernel'):
                    result_layer.depthwise_kernel = original_float_layer.depthwise_kernel
                if hasattr(original_float_layer, 'bias'):
                    result_layer.bias = original_float_layer.bias
                return result_layer
            return layer_to_substitue

        q_model = tf.keras.models.clone_model(quantized_model,
                                              clone_function=_disable_weights_quantization,
                                              input_tensors=quantized_model.inputs)
        return q_model, layers_name_mapping

    def disable_activations_quantization(self, float_model, quantized_model):

        def _disable_activation_quantization(layer_to_substitue):
            if isinstance(layer_to_substitue, KerasActivationQuantizationHolder):
                return keras.layers.Identity()
            return layer_to_substitue

        q_model = tf.keras.models.clone_model(quantized_model,
                                              clone_function=_disable_activation_quantization,
                                              input_tensors=quantized_model.inputs
                                              )

        return q_model
