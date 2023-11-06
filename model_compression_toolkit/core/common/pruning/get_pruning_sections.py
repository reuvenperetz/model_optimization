import numpy as np
from typing import List, Tuple, Dict

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.pruning.prunable_nodes import get_prunable_nodes
from model_compression_toolkit.core.common.pruning.pruning_section import PruningSection


def get_pruning_sections(graph, fw_info) -> List[PruningSection]:
    prunable_nodes = get_prunable_nodes(graph, fw_info)
    pruning_sections = []

    # Iterate over each prunable node and find its corresponding section
    for prunable_node in prunable_nodes:
        conv_nodes, intermediate_nodes = get_section_nodes(prunable_node, graph, fw_info)
        pruning_sections.append(PruningSection(conv_nodes, intermediate_nodes))

    return pruning_sections


def get_section_nodes(start_node: BaseNode,
                      graph,
                      fw_info) -> Tuple[List[BaseNode], List[BaseNode]]:

    intermediate_nodes = []

    # Follow the graph from the start_node to find the section's end
    next_node = graph.out_edges(start_node)[0].sink_node
    while not fw_info.is_kernel_op(next_node.type):
        intermediate_nodes.append(next_node)
        # Move to the next node in the section
        next_node = graph.out_edges(next_node)[0].sink_node

    assert fw_info.is_kernel_op(next_node.type)
    conv_nodes = [start_node, next_node]

    return conv_nodes, intermediate_nodes
