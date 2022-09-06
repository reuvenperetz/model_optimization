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
from model_compression_toolkit.core.pytorch.constants import KERNEL

from model_compression_toolkit.core.pytorch.utils import to_torch_tensor

from model_compression_toolkit import RoundingType, GradientPTQConfig
from model_compression_toolkit.core.common import BaseNode, Logger

from model_compression_toolkit.core.pytorch.back2framework.instance_builder import node_builder
from model_compression_toolkit.core.pytorch.back2framework.quantization_wrapper.quantized_layer_wrapper import \
    QuantizedLayerWrapper
from model_compression_toolkit.core.pytorch.back2framework.quantization_wrapper.wrapper_quantize_config import \
    WrapperQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.pytorch.wrappers_quantize_configs\
    .no_quantization_quantize_config import \
    NoQuantizationQuantizeConfig
from model_compression_toolkit.gptq.pytorch.quantizer.ste_rounding.ste_weights_quantizer import STEWeightQuantizer
from model_compression_toolkit.gptq.pytorch.weights_quantize_config import WeightsQuantizeConfig


def get_quantization_config(node: BaseNode, gptq_config: GradientPTQConfig) -> WrapperQuantizeConfig:
    """
    Create a QuantizeConfig to wrap a layer for its corresponding node.

    Args:
        node: Node to create a QuantizeConfig for.

    Returns:
        QuantizeConfig to use for wrapping the layer from the passed node.
    """

    if node.is_weights_quantization_enabled() and not node.is_activation_quantization_enabled():
        if not gptq_config.rounding_type == RoundingType.STE:
            Logger.critical('No support for GumbelRounding GPTQ yet. Work in progress..')
        else:
            float_weight_shape = to_torch_tensor(node.get_weights_by_keys(KERNEL)[0]).shape
            return WeightsQuantizeConfig([STEWeightQuantizer(node.final_weights_quantization_cfg,
                                                             gptq_config,
                                                             float_weight_shape)])

    # No quantization
    return NoQuantizationQuantizeConfig()
