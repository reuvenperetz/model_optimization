import numpy as np
from typing import List, Dict

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.pruning.get_pruning_sections import get_pruning_sections
from model_compression_toolkit.core.common.pruning.prune_graph import build_pruned_graph
from model_compression_toolkit.core.common.pruning.pruning_section import PruningSection, PruningSectionMask, \
    get_node_num_params


def get_pruned_graph_memory(graph,
                            masks:Dict[BaseNode,np.ndarray],
                            fw_info):
    total_memory = 0.
    pruning_sections = get_pruning_sections(graph, fw_info)
    nodes_to_prune = _get_all_nodes_to_prune(pruning_sections)
    for n in graph.nodes:
        if n not in nodes_to_prune:
            total_memory+=n.get_float_memory_bytes(fw_info)
    for pruning_section in pruning_sections:
        # If first node is the last node of some other pruning section
        # it means some of its input channels are removed
        first_node_input_channels_mask = None
        for other_section in pruning_sections:
            if pruning_section.conv_nodes[0]==other_section.conv_nodes[1]:
                first_node_input_channels_mask = masks[other_section.conv_nodes[0]]
                break

        # If second conv have some output channels that are removed
        second_node_output_mask = None
        if pruning_section.conv_nodes[1] in masks:
            second_node_output_mask = masks[pruning_section.conv_nodes[1]]

        pruning_section_mask = PruningSectionMask(first_node_input_mask=first_node_input_channels_mask,
                                                  first_node_output_mask=masks[pruning_section.conv_nodes[0]],
                                                  second_node_output_mask=second_node_output_mask)

        total_memory+=pruning_section.compute_memory(pruning_section_mask=pruning_section_mask,
                                                     fw_info=fw_info)

    shared_nodes = _get_nodes_from_adjacent_sections(pruning_sections)
    for n in shared_nodes:
        node_input_mask = get_node_input_mask(n, pruning_sections, masks)
        node_output_mask = masks[n] if n in masks else None
        total_memory -= get_node_num_params(node=n,
                                            input_mask=node_input_mask,
                                            output_mask=node_output_mask,
                                            fw_info=fw_info)*4.
    return total_memory

def get_node_input_mask(node, pruning_sections, masks):
    input_mask = None
    for section in pruning_sections:
        if node==section.conv_nodes[1]:
            input_mask = masks[section.conv_nodes[0]]
            break
    return input_mask

def _get_nodes_from_adjacent_sections(pruning_sections: List[PruningSection]):
    """
    Get a list of nodes that are the last node of one pruning section and the first node of
    another section.

    Args:
        pruning_sections: List of PruningSection objects representing sections of a neural network.

    Returns:
        List of BaseNode objects that are shared between adjacent sections.
    """
    # Create a set for the first and last nodes of each section
    first_nodes = set(section.conv_nodes[0] for section in pruning_sections if section.conv_nodes)
    last_nodes = set(section.conv_nodes[-1] for section in pruning_sections if section.conv_nodes)

    # Find nodes that are both, the first in one section and last in another
    shared_nodes = first_nodes.intersection(last_nodes)

    return list(shared_nodes)



    # pruned_graph = build_pruned_graph(graph,
    #                                   masks,
    #                                   fw_info,
    #                                   fw_impl)
    # return sum([n.get_float_memory_bytes(fw_info) for n in pruned_graph.nodes])


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
