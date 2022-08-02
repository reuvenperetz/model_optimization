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

from typing import List, Tuple, Any

from tensorflow import Tensor
import tensorflow as tf
# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer

if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers import Layer
else:
    from keras.engine.base_layer import Layer
from tensorflow.python.training.tracking.data_structures import ListWrapper
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig



class WeightsQuantizeConfig(QuantizeConfig):

    def __init__(self,
                 w_quantizer: Quantizer,
                 weight_attrs: List[str] = None):

        self.weight_attrs = weight_attrs
        self.w_quantizer = w_quantizer

    def get_config(self):
        return {'w_quantizer': self.w_quantizer,
                'weight_attrs': self.weight_attrs}

    def get_weights_and_quantizers(self, layer: Layer) -> List[Tuple[Tensor, Any]]:
        return [(getattr(layer, self.weight_attrs[i]),
                 self.w_quantizer) for i in range(len(self.weight_attrs))]

    def get_activations_and_quantizers(self, layer: Layer) -> list:
        # For configurable activations we use get_output_quantizers,
        # Therefore, we do not need to implement this method.
        return []

    def set_quantize_weights(self, layer: Layer, quantize_weights: List[Tensor]):
        if len(self.weight_attrs) != len(quantize_weights):
            raise ValueError(
                '`set_quantize_weights` called on layer {} with {} '
                'weight parameters, but layer expects {} values.'.format(
                    layer.name, len(quantize_weights), len(self.weight_attrs)))

        for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
            current_weight = getattr(layer, weight_attr)
            if current_weight.shape != weight.shape:
                raise ValueError('Existing layer weight shape {} is incompatible with'
                                 'provided weight shape {}'.format(
                    current_weight.shape, weight.shape))

            setattr(layer, weight_attr, weight)

    def set_quantize_activations(self, layer, quantize_activations: ListWrapper):
        pass

    def get_output_quantizers(self, layer: Layer) -> list:
        return []






class ActivationQuantizeConfig(QuantizeConfig):

    def __init__(self,
                 activation_quantizer: Quantizer,
                 # activation_attrs: List[str] = None
                 ):

        self.activation_quantizer = activation_quantizer
        # self.activation_attrs = activation_attrs

    def get_config(self):
        return {
            # 'activation_attrs': self.activation_attrs,
                'activation_quantizer': self.activation_quantizer}

    def get_weights_and_quantizers(self, layer: Layer) -> List[Tuple[Tensor, Any]]:
        return []

    def get_activations_and_quantizers(self, layer: Layer) -> list:
        # For configurable activations we use get_output_quantizers,
        # Therefore, we do not need to implement this method.
        return []

    def set_quantize_weights(self, layer: Layer, quantize_weights: List[Tensor]):
        pass

    def set_quantize_activations(self, layer, quantize_activations: ListWrapper):
        pass
        # if len(self.activation_attrs) != len(quantize_activations):
        #     raise ValueError(
        #         '`set_quantize_weights` called on layer {} with {} '
        #         'weight parameters, but layer expects {} values.'.format(
        #             layer.name, len(quantize_activations), len(self.activation_attrs)))
        #
        # for activation_attr, quantized_activation in zip(self.activation_attrs, quantize_activations):
        #     current_activation = getattr(layer, activation_attr)
        #     if current_activation.shape != quantized_activation.shape:
        #         raise ValueError('Existing layer weight shape {} is incompatible with'
        #                          'provided weight shape {}'.format(
        #             current_activation.shape, quantized_activation.shape))
        #
        #     setattr(layer, activation_attr, quantized_activation)

    def get_output_quantizers(self, layer: Layer) -> list:
        return [self.activation_quantizer]




class WeightsActivationQuantizeConfig(QuantizeConfig):

    def __init__(self,
                 activation_quantizer: Quantizer,
                 w_quantizer: Quantizer,
                 weight_attrs: List[str] = None
                 ):

        self.act_config = ActivationQuantizeConfig(activation_quantizer=activation_quantizer)
        self.weights_config = WeightsQuantizeConfig(w_quantizer=w_quantizer,
                                                    weight_attrs=weight_attrs)

    def get_config(self):
        return {"activation_quantizer": self.act_config.activation_quantizer,
                "w_quantizer": self.weights_config.w_quantizer,
                "weight_attrs": self.weights_config.weight_attrs}

    def get_weights_and_quantizers(self, layer: Layer) -> List[Tuple[Tensor, Any]]:
        return self.weights_config.get_weights_and_quantizers(layer)

    def get_activations_and_quantizers(self, layer: Layer) -> list:
        return self.act_config.get_activations_and_quantizers(layer)

    def set_quantize_weights(self, layer: Layer, quantize_weights: List[Tensor]):
        self.weights_config.set_quantize_weights(layer, quantize_weights)

    def set_quantize_activations(self, layer, quantize_activations: ListWrapper):
        self.act_config.set_quantize_activations(layer, quantize_activations)

    def get_output_quantizers(self, layer: Layer) -> list:
        return self.act_config.get_output_quantizers(layer)
