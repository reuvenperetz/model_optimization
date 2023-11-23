import numpy as np
from typing import List, Dict

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.pruning.pruning_section import PruningSection, PruningSectionMask


class MemoryCalculator:
    def __init__(self,
                 graph: Graph,
                 fw_info: FrameworkInfo):
        self.graph = graph
        self.fw_info = fw_info

    def get_pruned_graph_memory(self, masks: Dict[BaseNode, np.ndarray]):
        total_memory = 0.
        pruning_sections = self.graph.get_pruning_sections(self.fw_info)

        nodes_to_prune = []
        for section in pruning_sections:
            nodes_to_prune.append(section.input_node)
            nodes_to_prune.extend(section.intermediate_nodes)
            nodes_to_prune.append(section.output_node)

        for n in self.graph.nodes:
            if n not in nodes_to_prune:
                total_memory += n.get_float_memory_bytes(self.fw_info)

        for pruning_section in pruning_sections:
            # If first node is the last node of some other pruning section
            # it means some of its input channels are removed
            first_node_input_channels_mask = None
            for other_section in pruning_sections:
                if pruning_section.input_node == other_section.output_node:
                    first_node_input_channels_mask = masks[other_section.input_node]
                    break

            # If second conv have some output channels that are removed
            second_node_output_mask = None
            if pruning_section.output_node in masks:
                second_node_output_mask = masks[pruning_section.output_node]

            pruning_section_mask = PruningSectionMask(input_node_ic_mask=first_node_input_channels_mask,
                                                      input_node_oc_mask=masks[pruning_section.input_node],
                                                      output_node_oc_mask=second_node_output_mask)

            total_memory += self._get_pruning_section_memory(pruning_section,
                                                             pruning_section_mask)

        shared_nodes = self._get_nodes_from_adjacent_sections(pruning_sections)
        for n in shared_nodes:
            node_input_mask = self._get_node_input_mask(n, pruning_sections, masks)
            node_output_mask = masks[n] if n in masks else None
            total_memory -= self._get_node_num_params(node=n,
                                                      input_mask=node_input_mask,
                                                      output_mask=node_output_mask) * 4.
        return total_memory

    def _get_node_input_mask(self, node, pruning_sections, masks):
        input_mask = None
        for section in pruning_sections:
            if node == section.output_node:
                input_mask = masks[section.input_node]
                break
        return input_mask

    def _get_nodes_from_adjacent_sections(self, pruning_sections: List[PruningSection]):
        """
        Get a list of nodes that are the last node of one pruning section and the first node of
        another section.

        Args:
            pruning_sections: List of PruningSection objects representing sections of a neural network.

        Returns:
            List of BaseNode objects that are shared between adjacent sections.
        """
        # Create a set for the first and last nodes of each section
        input_nodes = set(section.input_node for section in pruning_sections)
        output_nodes = set(section.output_node for section in pruning_sections)
        # Find nodes that are both, the first in one section and last in another
        shared_nodes = input_nodes.intersection(output_nodes)
        return list(shared_nodes)

    def _get_node_num_params(self,
                             node: BaseNode,
                             input_mask: np.ndarray,
                             output_mask: np.ndarray):
        total_params = 0
        if self.fw_info.is_kernel_op(node.type):
            oc_axis, ic_axis = self.fw_info.kernel_channels_mapping.get(node.type)
            kernel_attr = self.fw_info.get_kernel_op_attributes(node.type)[0]
            for w_attr, w in node.weights.items():
                if kernel_attr in w_attr:
                    # Convert masks to boolean arrays, treating None as all ones
                    input_mask = np.ones(w.shape[ic_axis], dtype=bool) if input_mask is None else input_mask.astype(
                        bool)
                    output_mask = np.ones(w.shape[oc_axis], dtype=bool) if output_mask is None else output_mask.astype(
                        bool)

                    # Apply masks to the kernel using np.take
                    pruned_w = np.take(w, np.where(input_mask)[0], axis=ic_axis)
                    pruned_w = np.take(pruned_w, np.where(output_mask)[0], axis=oc_axis)

                    # Calculate the number of remaining parameters
                    total_params += len(pruned_w.flatten())
                else:
                    total_params += len(np.take(w, np.where(output_mask)[0]))

        else:
            for w_attr, w in node.weights.items():
                total_params += len(np.take(w, np.where(output_mask)[0]))

        return total_params

    def _get_pruning_section_memory(self,
                                    pruning_section: PruningSection,
                                    pruning_section_mask: PruningSectionMask):
        """
        Computes the memory footprint of the section, considering the weights
        of the convolutional nodes and the size of intermediate data.
        """
        first_node_memory = self._get_node_num_params(node=pruning_section.input_node,
                                                      input_mask=pruning_section_mask.input_node_ic_mask,
                                                      output_mask=pruning_section_mask.input_node_oc_mask)
        total_inter_nodes_memory = 0
        for inter_node in pruning_section.intermediate_nodes:
            total_inter_nodes_memory += self._get_node_num_params(node=inter_node,
                                                                  input_mask=pruning_section_mask.input_node_oc_mask,
                                                                  output_mask=pruning_section_mask.input_node_oc_mask)

        second_node_memory = self._get_node_num_params(node=pruning_section.output_node,
                                                       input_mask=pruning_section_mask.input_node_oc_mask,
                                                       output_mask=pruning_section_mask.output_node_oc_mask)

        return (first_node_memory + total_inter_nodes_memory + second_node_memory) * 4.
