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
from typing import Dict, Callable

import keras

import mct_quantizers
from mct_quantizers import KerasActivationQuantizationHolder, KerasQuantizationWrapper
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo

from model_compression_toolkit.core.keras.reader.reader import model_reader
from model_compression_toolkit.logger import Logger

from model_compression_toolkit.xquant.common.constants import XQUANT_REPR, INTERMEDIATE_SIMILARITY_METRICS_REPR, \
    XQUANT_VAL, INTERMEDIATE_SIMILARITY_METRICS_VAL
from model_compression_toolkit.xquant.common.tensorboard_utils import TensorboardUtils

NODES_WITHOUT_CUT_INFO = [KerasActivationQuantizationHolder]


class KerasTensorboardUtils(TensorboardUtils):
    """
    A utility class for handling TensorBoard operations specific to Keras models.
    This class extends the generic TensorboardUtils class and provides methods
    to facilitate the visualization of quantized models and their similarity metrics
    in TensorBoard.
    """

    def __init__(self, report_dir: str,
                 fw_info: FrameworkInfo,
                 fw_impl: FrameworkImplementation):
        """
        Initialize the KerasTensorboardUtils class with the given parameters.

        Args:
            report_dir (str): Directory where the TensorBoard files will be stored.
            fw_info (FrameworkInfo): Information about the framework being used.
            fw_impl (FrameworkImplementation): Implementation functions for the framework.
        """
        super().__init__(report_dir,
                         fw_info,
                         fw_impl)

    def get_graph_for_tensorboard_display(self,
                                          quantized_model: keras.Model,
                                          similarity_metrics: Dict[str, Dict[str, float]],
                                          repr_dataset: Callable) -> Graph:
        """
        Generate a graph suitable for TensorBoard display from the provided quantized model
        and similarity metrics.

        Args:
            quantized_model (keras.Model): The quantized Keras model for which the graph is to be created.
            similarity_metrics (Dict[str, Dict[str, float]]): A dictionary containing similarity metrics
                for different nodes in the model.
            repr_dataset (Callable): A function or callable that provides the representative dataset.

        Returns:
            Graph: A graph object representing the quantized model, annotated with similarity metrics.
        """
        # Read the quantized model into a graph structure.
        quant_graph = model_reader(quantized_model)
        from mct_quantizers.keras.metadata import get_metadata
        _metadata = get_metadata(quantized_model)

        fused_node_to_memory_elements = {}
        for cut in _metadata['scheduling_info']['cuts']:
            fused_node = cut['op_order'][-1]
            if not fused_node.startswith('DummyType'):
                fused_node_to_memory_elements[fused_node] = cut['mem_elements']

        for node in quant_graph.nodes:
            if node.type not in NODES_WITHOUT_CUT_INFO:
                if node.name in fused_node_to_memory_elements:
                    node.framework_attr['cut_memory_elements'] = [
                        f"{mem_element['node_name']}_outTensor_{mem_element['node_output_index']}" for mem_element in
                        fused_node_to_memory_elements[node.name]]
                elif node.type == KerasQuantizationWrapper and node.framework_attr['layer']['config'][
                    'name'] in fused_node_to_memory_elements:
                    node.framework_attr['cut_memory_elements'] = [
                        f"{mem_element['node_name']}_outTensor_{mem_element['node_output_index']}" for mem_element in
                        fused_node_to_memory_elements[node.framework_attr['layer']['config']['name']]]

                elif node.name in _metadata['scheduling_info']['fused_nodes_mapping']:
                    node.framework_attr['cut_memory_elements'] = [
                        f"{mem_element['node_name']}_outTensor_{mem_element['node_output_index']}" for mem_element in
                        fused_node_to_memory_elements[_metadata['scheduling_info']['fused_nodes_mapping'][node.name]]]
                elif node.type == KerasQuantizationWrapper and node.framework_attr['layer']['config']['name'] in \
                        _metadata['scheduling_info']['fused_nodes_mapping']:
                    node.framework_attr['cut_memory_elements'] = [
                        f"{mem_element['node_name']}_outTensor_{mem_element['node_output_index']}" for mem_element in
                        fused_node_to_memory_elements[_metadata['scheduling_info']['fused_nodes_mapping'][
                            node.framework_attr['layer']['config']['name']]]]
                #
                # elif node.type == KerasQuantizationWrapper and node.framework_attr['layer']['config']['name'] in
                # fused_node_to_memory_elements:
                #     node.framework_attr['cut_memory_elements'] = [
                #         f"{mem_element['node_name']}_outTensor_{mem_element['node_output_index']}" for mem_element in
                #         fused_node_to_memory_elements[fused_node_to_memory_elements[node.framework_attr['layer'][
                #         'config']['name']]]]

                else:
                    Logger.critical(f"Node {node.name} has no cut tensors information")

        # Iterate over each node in the graph.
        for node in quant_graph.nodes:
            # Check if the node's name is in the similarity metrics for intermediate representation.
            if node.name in similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_REPR].keys():
                # If so, add the similarity metric for intermediate representation to the node's attributes.
                node.framework_attr[XQUANT_REPR] = similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_REPR][node.name]

            # Check if the node's name is in the similarity metrics for validation.
            if node.name in similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_VAL].keys():
                # If so, add the similarity metric for validation to the node's attributes.
                node.framework_attr[XQUANT_VAL] = similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_VAL][node.name]

        return quant_graph
