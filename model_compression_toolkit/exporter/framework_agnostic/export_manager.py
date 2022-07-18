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


from typing import Tuple, Any, List

from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.quantization.quantize_graph_weights import quantize_graph_weights
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter


class ExportManager:

    def __init__(self,
                 graph: Graph,
                 fw_impl: FrameworkImplementation,
                 fw_info: FrameworkInfo,
                 tb_writer: TensorboardWriter,
                 bit_widths_config: List[int],
                 export_target: Any):

        self.graph = graph
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.export_target = export_target
        self.tb_writer = tb_writer
        self.bit_widths_config = bit_widths_config

    def export_model(self):
        """
        A function for quantizing the graph's weights and build a quantized framework model from it.

        Args:
            tg: A prepared for quantization graph.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
            groups of layers by how they should be quantized, etc.).
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.
            tb_w: TensorBoardWriter object to log events.
            bit_widths_config: mixed-precision bit configuration to be added to model user_info

        Returns:
            Quantized model in the input framework, and information the user may need in order to use the quantized
            model.
        """
        quantized_model, user_info = self._quantize_model()
        user_info.mixed_precision_cfg = self.bit_widths_config
        return quantized_model, user_info


    def _quantize_model(self) -> Tuple[Any, UserInformation]:
        """
        Quantize graph's weights, and build a quantized framework model from it.

        Args:
            tg: A prepared for quantization graph.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.).
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.
            tb_w: TensorBoardWriter object to log events.

        Returns:
            Quantized model in the input framework, and information the user may need in order to use the quantized model.
        """

        quantized_tg = quantize_graph_weights(self.graph,
                                              fw_info=self.fw_info,
                                              fw_impl=self.fw_impl)
        if self.tb_writer is not None:
            self.tb_writer.add_graph(quantized_tg, 'after_quantization')

        ######################################
        # Back2Framewor
        ######################################
        # Before building a quantized model, first apply some substitutions.
        quantized_tg = substitute(quantized_tg,
                                  self.fw_impl.get_substitutions_pre_build())

        quantized_model, user_info = self.fw_impl.model_builder(quantized_tg,
                                                           mode=ModelBuilderMode.QUANTIZED,
                                                           fw_info=self.fw_info)
        return quantized_model, user_info

