# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Dict, Any

from model_compression_toolkit.core.common import BaseNode, Logger
from model_compression_toolkit.core.common.constants import THRESHOLD, RANGE_MIN, RANGE_MAX, SIGNED
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure import keras_inferable_quantizers
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers.base_keras_inferable_quantizer \
    import \
    BaseKerasInferableQuantizer

QUANTIZATION_METHOD_2_WEIGHTS_QUANTIZER = {
    QuantizationMethod.POWER_OF_TWO: keras_inferable_quantizers.WeightsPOTInferableQuantizer,
    QuantizationMethod.SYMMETRIC: keras_inferable_quantizers.WeightsSymmetricInferableQuantizer,
    QuantizationMethod.UNIFORM: keras_inferable_quantizers.WeightsUniformInferableQuantizer
}

QUANTIZATION_METHOD_2_ACTIVATION_QUANTIZER = {
    QuantizationMethod.POWER_OF_TWO: keras_inferable_quantizers.ActivationPOTInferableQuantizer,
    QuantizationMethod.SYMMETRIC: keras_inferable_quantizers.ActivationSymmetricInferableQuantizer,
    QuantizationMethod.UNIFORM: keras_inferable_quantizers.ActivationUniformInferableQuantizer
}


def get_inferable_quantizer_kwargs(node: BaseNode,
                                   quantization_target: QuantizationTarget) -> Dict[str, Any]:
    """
    Get the quantization parameters for an inferable quantizer.
    Args:
        node: The node for which the quantizer is being created.
        quantization_target: The target of the quantization (weights or activations).

    Returns:
        The quantization parameters as a dictionary.

    """

    if quantization_target == QuantizationTarget.Weights:
        # Get the weights quantization configuration for the node
        node_w_qc = node.final_weights_quantization_cfg
        quantization_method = node_w_qc.weights_quantization_method

        # Check if the quantization method is supported for inferable quantizers
        assert quantization_method in QUANTIZATION_METHOD_2_WEIGHTS_QUANTIZER, f'{quantization_method} for weights ' \
                                                                               f'not in supported quantization ' \
                                                                               f'methods for inferable quantizers'

        # Return the appropriate quantization parameters based on the quantization method
        if quantization_method in [QuantizationMethod.POWER_OF_TWO,
                                   QuantizationMethod.SYMMETRIC]:
            return {'num_bits': node_w_qc.weights_n_bits,
                    'threshold': node_w_qc.weights_quantization_params.get(THRESHOLD),
                    'signed': True,
                    'per_channel': node_w_qc.weights_per_channel_threshold,
                    'channel_axis': node_w_qc.weights_channels_axis,
                    'input_num_dims': node_w_qc.weights_quantization_params.get(THRESHOLD).ndim}

        elif quantization_method in [QuantizationMethod.UNIFORM]:
            return {'num_bits': node_w_qc.weights_n_bits,
                    'per_channel': node_w_qc.weights_per_channel_threshold,
                    'min_range': node_w_qc.weights_quantization_params.get(RANGE_MIN),
                    'max_range': node_w_qc.weights_quantization_params.get(RANGE_MAX),
                    'input_num_dims': node_w_qc.weights_quantization_params.get(THRESHOLD).ndim}
        else:
            Logger.critical(f'Not supported quantization method for inferable quantizers.')  # pragma: no cover

    elif quantization_target == QuantizationTarget.Activation:
        # Get the activation quantization configuration for the node
        node_qc = node.final_activation_quantization_cfg
        quantization_method = node_qc.activation_quantization_method

        # Check if the quantization method is supported for inferable quantizers
        assert quantization_method in QUANTIZATION_METHOD_2_ACTIVATION_QUANTIZER, f'{quantization_method} for weights ' \
                                                                                  f'not in ' \
                                                                                  f'supported quantization methods ' \
                                                                                  f'for inferable' \
                                                                                  f' quantizers'

        # Return the appropriate quantization parameters based on the quantization method
        if quantization_method in [QuantizationMethod.POWER_OF_TWO,
                                   QuantizationMethod.SYMMETRIC]:
            return {'num_bits': node_qc.activation_n_bits,
                    'threshold': node_qc.activation_quantization_params.get(THRESHOLD),
                    'signed': node_qc.activation_quantization_params.get(SIGNED)}

        elif quantization_method in [QuantizationMethod.UNIFORM]:
            return {'num_bits': node_qc.activation_n_bits,
                    'min_range': node_qc.activation_quantization_params.get(RANGE_MIN),
                    'max_range': node_qc.activation_quantization_params.get(RANGE_MAX)}
        else:
            Logger.critical(f'Not supported quantization method for inferable quantizers.')  # pragma: no cover
    else:
        Logger.critical(f'{quantization_target} is not supported')  # pragma: no cover



def get_weights_quantizer_for_node(node: BaseNode) -> BaseKerasInferableQuantizer:
    """
    Get weights quantizer for a node.

    Args:
        node: Node to create a weight quantizer for.

    Returns:
        Quantizer for the node's weights.

    """
    if node.final_weights_quantization_cfg is None:
        Logger.critical(f'Can not set quantizer for a node with no final weights quantization configuration')  # pragma:
        # no cover
    node_w_qc = node.final_weights_quantization_cfg
    weights_quantization_method = node_w_qc.weights_quantization_method
    kwargs = get_inferable_quantizer_kwargs(node, QuantizationTarget.Weights)
    return QUANTIZATION_METHOD_2_WEIGHTS_QUANTIZER.get(weights_quantization_method)(**kwargs)


def get_activations_quantizer_for_node(node: BaseNode) -> BaseKerasInferableQuantizer:
    """
    Get activation quantizer for a node.

    Args:
        node: Node to create an activation quantizer for.

    Returns:
        Quantizer for the node's activations.

    """
    if node.final_activation_quantization_cfg is None:
        Logger.critical(f'Can not set quantizer for a node with no final activation quantization configuration')  #
        # pragma: no cover
    node_act_qc = node.final_activation_quantization_cfg
    activation_quantization_method = node_act_qc.activation_quantization_method
    kwargs = get_inferable_quantizer_kwargs(node, QuantizationTarget.Activation)
    return QUANTIZATION_METHOD_2_ACTIVATION_QUANTIZER.get(activation_quantization_method)(**kwargs)
