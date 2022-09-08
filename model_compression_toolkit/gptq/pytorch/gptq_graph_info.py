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
import torch.nn as nn
from typing import List

from model_compression_toolkit.core.pytorch.back2framework.quantization_wrapper.quantized_layer_wrapper import \
    QuantizedLayerWrapper
from model_compression_toolkit.gptq.pytorch.quantizer.quantizer_wrapper import WeightQuantizerWrapper
from model_compression_toolkit.gptq.pytorch.weights_quantize_config import WeightsQuantizeConfig


def get_trainable_parameters(fxp_model: nn.Module,
                             add_bias: bool = False,
                             quantization_parameters_learning: bool = False) -> (List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]):
    """
    Get trainable parameters from all layers in a model

    Args:
        fxp_model: Model to get its trainable parameters.
        add_bias: Whether to include biases of the model (if there are) or not.
        quantization_parameters_learning: Whether to include quantization parameters of the model or not.
    Returns:
        A list of trainable variables in a model. Each item is a list of a layers weights.
    """

    trainable_aux_weights = nn.ParameterList()
    trainable_threshold = nn.ParameterList()
    trainable_bias = nn.ParameterList()

    for layer in fxp_model.modules():
        if isinstance(layer, QuantizedLayerWrapper) and isinstance(layer.quantization_config, WeightsQuantizeConfig):
            trainable_aux_weights.append(layer.quantization_config.get_weight_quantizers()[0].get_aux_variable())
            if quantization_parameters_learning:
                trainable_threshold.append(layer.quantization_config.get_weight_quantizers()[0].get_quantization_variable())
            if add_bias and layer.layer.bias is not None:
                trainable_bias.append(layer.layer.bias)

    return trainable_aux_weights, trainable_bias, trainable_threshold


def get_weights_for_loss(fxp_model: nn.Module) -> [List, List]:
    """
    Get all float and quantized kernels for the GPTQ loss

    Args:
        fxp_model: Model to get its float and quantized weights.

    Returns:
        A list of float kernels, each item is the float kernel of the layer
        A list of quantized kernels, each item is the quantized kernel of the layer
    """

    flp_weights_list, fxp_weights_list = [], []
    for layer in fxp_model.modules():
        if isinstance(layer, QuantizedLayerWrapper) and isinstance(layer.quantization_config, WeightsQuantizeConfig):
            # Collect pairs of float and quantized weights per layer
            weights = layer.layer.weight
            flp_weights_list.append(weights)
            fxp_weights_list.append(layer.quantization_config.get_weight_quantizers()[0](weights))

    return flp_weights_list, fxp_weights_list
