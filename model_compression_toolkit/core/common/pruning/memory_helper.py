from typing import Dict, List

from model_compression_toolkit.core.common import BaseNode
import numpy as np

# from model_compression_toolkit.core.common.pruning.graph_helper import GraphHelper, PruningSection, PruningSectionMask


class MemoryHelper:

    @staticmethod
    def get_pruned_graph_memory(graph,
                                masks: Dict[BaseNode, np.ndarray],
                                fw_info):
        total_memory = 0.
        pruning_sections = GraphHelper.get_pruning_sections(graph, fw_info)
        nodes_to_prune = _get_all_nodes_to_prune(pruning_sections)
        for n in graph.nodes:
            if n not in nodes_to_prune:
                total_memory += n.get_float_memory_bytes(fw_info)
        for pruning_section in pruning_sections:
            # If first node is the last node of some other pruning section
            # it means some of its input channels are removed
            first_node_input_channels_mask = None
            for other_section in pruning_sections:
                if pruning_section.conv_nodes[0] == other_section.conv_nodes[1]:
                    first_node_input_channels_mask = masks[other_section.conv_nodes[0]]
                    break

            # If second conv have some output channels that are removed
            second_node_output_mask = None
            if pruning_section.conv_nodes[1] in masks:
                second_node_output_mask = masks[pruning_section.conv_nodes[1]]

            pruning_section_mask = PruningSectionMask(first_node_input_mask=first_node_input_channels_mask,
                                                      first_node_output_mask=masks[pruning_section.conv_nodes[0]],
                                                      second_node_output_mask=second_node_output_mask)

            total_memory += pruning_section.compute_memory(pruning_section_mask=pruning_section_mask,
                                                           fw_info=fw_info)

        shared_nodes = GraphHelper.get_nodes_from_adjacent_sections(pruning_sections)
        for n in shared_nodes:
            node_input_mask = get_node_input_mask(n, pruning_sections, masks)
            node_output_mask = masks[n] if n in masks else None
            total_memory -= MemoryHelper.get_node_num_params(node=n,
                                                             input_mask=node_input_mask,
                                                             output_mask=node_output_mask,
                                                             fw_info=fw_info) * 4.
        return total_memory

    @staticmethod
    def get_node_num_params(node: BaseNode,
                            input_mask: np.ndarray,
                            output_mask: np.ndarray,
                            fw_info):
        total_params = 0
        if fw_info.is_kernel_op(node.type):
            oc_axis, ic_axis = fw_info.kernel_channels_mapping.get(node.type)
            kernel_attr = fw_info.get_kernel_op_attributes(node.type)[0]
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


def _get_all_nodes_to_prune(pruning_sections: List[PruningSection]) -> List:
    """
    Returns a list of all nodes contained within the given pruning sections.

    :param pruning_sections: A list of PruningSection objects.
    :return: A list of all nodes to be pruned.
    """
    nodes_to_prune = []
    for section in pruning_sections:
        # Add the convolutional and intermediate nodes from the section to the list
        nodes_to_prune.extend(section.conv_nodes)
        nodes_to_prune.extend(section.intermediate_nodes)

    return nodes_to_prune


def get_node_input_mask(node, pruning_sections, masks):
    input_mask = None
    for section in pruning_sections:
        if node == section.conv_nodes[1]:
            input_mask = masks[section.conv_nodes[0]]
            break
    return input_mask
