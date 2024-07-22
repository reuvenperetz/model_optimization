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
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchQuantizationWrapper
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from typing import Dict, Any, Callable

import torch

from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.pytorch.reader.reader import model_reader
from model_compression_toolkit.xquant.common.constants import XQUANT_REPR, INTERMEDIATE_SIMILARITY_METRICS_REPR, XQUANT_VAL, INTERMEDIATE_SIMILARITY_METRICS_VAL
from model_compression_toolkit.xquant.common.model_folding_utils import ModelFoldingUtils
from model_compression_toolkit.xquant.common.tensorboard_utils import TensorboardUtils

NODES_WITHOUT_CUT_INFO = [torch.fake_quantize_per_tensor_affine]

def is_wrapped_linear_op(quantized_model, node):
    # Check if a node in a torch fx graph represents a linear layer (conv2d/linear)
    # that is wrapped in the quantized model
    return hasattr(quantized_model, node.name.removesuffix('_layer')) and isinstance(
        getattr(quantized_model, node.name.removesuffix('_layer')), PytorchQuantizationWrapper)

class PytorchTensorboardUtils(TensorboardUtils):
    """
    Utility class for handling PyTorch models with TensorBoard. Inherits from TensorboardUtils.
    This class provides functionalities to display quantized model graphs on TensorBoard.
    """

    def __init__(self,
                 report_dir: str,
                 fw_info: FrameworkInfo,
                 fw_impl: FrameworkImplementation):
        """
        Initialize the PytorchTensorboardUtils instance.

        Args:
            report_dir: Directory where the reports are stored.
            fw_info: Information about the framework being used.
            fw_impl: Implementation methods for the framework.
        """
        super().__init__(report_dir,
                         fw_info,
                         fw_impl)

    def get_graph_for_tensorboard_display(self,
                                          quantized_model: torch.nn.Module,
                                          similarity_metrics: Dict[str, Any],
                                          repr_dataset: Callable):
        """
        Get the graph to display on TensorBoard. The graph represents the quantized model
        with the similarity metrics that were measured.

        Args:
            quantized_model: The quantized model to be displayed on TensorBoard.
            similarity_metrics: Dictionary containing the collected similarity metrics values.
            repr_dataset: Callable that generates the representative dataset used during graph building.

        Returns:
            The updated quantized model graph with similarity metrics embedded.
        """
        # Read the model and generate a graph representation
        quant_graph = model_reader(quantized_model,
                                   representative_data_gen=repr_dataset,
                                   to_tensor=self.fw_impl.to_tensor,
                                   to_numpy=self.fw_impl.to_numpy)

        from mct_quantizers.pytorch.metadata import get_metadata
        _metadata = get_metadata(quantized_model)

        fused_node_to_memory_elements = {}
        for cut in _metadata['scheduling_info']['cuts']:
            fused_node = cut['op_order'][-1]
            if not fused_node.startswith('DummyType'):
                fused_node_to_memory_elements[fused_node] = cut['mem_elements']

        for node in quant_graph.nodes:
            if node.type not in NODES_WITHOUT_CUT_INFO:
                # Layer that was not part of some fusion should be in this dictionary:
                if node.name in fused_node_to_memory_elements:
                    node.framework_attr['cut_memory_elements'] = [f"{mem_element['node_name']}_outTensor_{mem_element['node_output_index']}" for mem_element in fused_node_to_memory_elements[node.name]]
                    node.framework_attr['cut_total_size'] = sum([mem_element['total_size'] for mem_element in fused_node_to_memory_elements[node.name]])

                # Conv2D/Linear that was not part of some fusion should be in this dictionary:
                elif is_wrapped_linear_op(quantized_model, node) and node.name.removesuffix('_layer') in fused_node_to_memory_elements:
                    node.framework_attr['cut_memory_elements'] = [f"{mem_element['node_name']}_outTensor_{mem_element['node_output_index']}" for mem_element in fused_node_to_memory_elements[node.name.removesuffix('_layer')]]
                    node.framework_attr['cut_total_size'] = sum([mem_element['total_size'] for mem_element in fused_node_to_memory_elements[node.name.removesuffix('_layer')]])

                elif node.name in _metadata['scheduling_info']['fused_nodes_mapping']:
                    node.framework_attr['cut_memory_elements'] = [f"{mem_element['node_name']}_outTensor_{mem_element['node_output_index']}" for mem_element in fused_node_to_memory_elements[_metadata['scheduling_info']['fused_nodes_mapping'][node.name]]]
                    node.framework_attr['cut_total_size'] = sum([mem_element['total_size'] for mem_element in fused_node_to_memory_elements[_metadata['scheduling_info']['fused_nodes_mapping'][node.name]]])

                elif is_wrapped_linear_op(quantized_model, node) and node.name.removesuffix('_layer') in _metadata['scheduling_info']['fused_nodes_mapping']:
                    node.framework_attr['cut_memory_elements'] = [f"{mem_element['node_name']}_outTensor_{mem_element['node_output_index']}" for mem_element in fused_node_to_memory_elements[_metadata['scheduling_info']['fused_nodes_mapping'][node.name.removesuffix('_layer')]]]
                    node.framework_attr['cut_total_size'] = sum([mem_element['total_size'] for mem_element in fused_node_to_memory_elements[_metadata['scheduling_info']['fused_nodes_mapping'][node.name.removesuffix('_layer')]]])

                else:
                    print(f"Node {node.name} has no cut tensors information")
                    # Logger.critical(f"Node {node.name} has no cut tensors information")

        # Iterate through each node in the graph
        for node in quant_graph.nodes:
            # Check and add similarity metrics for each node in the graph
            if node.name in similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_REPR].keys():
                node.framework_attr[XQUANT_REPR] = similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_REPR][f"{node.name}"]
            elif node.name.removesuffix("_layer") in similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_REPR].keys():
                node.framework_attr[XQUANT_REPR] = similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_REPR][
                    node.name.removesuffix("_layer")]

            # Check and add validation similarity metrics for each node in the graph
            if node.name in similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_VAL].keys():
                node.framework_attr[XQUANT_VAL] = similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_VAL][f"{node.name}"]
            elif node.name.removesuffix("_layer") in similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_VAL].keys():
                node.framework_attr[XQUANT_VAL] = similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_VAL][
                    node.name.removesuffix("_layer")]

        return quant_graph
