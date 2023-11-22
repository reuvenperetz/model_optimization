# from typing import List, Tuple
#
# from model_compression_toolkit.core.common import BaseNode
# from networkx.algorithms.dag import topological_sort
# import numpy as np
#
# # from model_compression_toolkit.core.common.pruning.memory_helper import MemoryHelper
#
#
# class GraphHelper:
#
#     # @staticmethod
#     def get_nodes_from_adjacent_sections(pruning_sections: List[PruningSection]):
#         """
#         Get a list of nodes that are the last node of one pruning section and the first node of
#         another section.
#
#         Args:
#             pruning_sections: List of PruningSection objects representing sections of a neural network.
#
#         Returns:
#             List of BaseNode objects that are shared between adjacent sections.
#         """
#         # Create a set for the first and last nodes of each section
#         first_nodes = set(section.conv_nodes[0] for section in pruning_sections if section.conv_nodes)
#         last_nodes = set(section.conv_nodes[-1] for section in pruning_sections if section.conv_nodes)
#
#         # Find nodes that are both, the first in one section and last in another
#         shared_nodes = first_nodes.intersection(last_nodes)
#
#         return list(shared_nodes)
#
#     # @staticmethod
#     def get_prunable_nodes(float_graph, fw_info) -> List[BaseNode]:
#         prunable_nodes = []
#         for n in list(topological_sort(float_graph)):
#             if GraphHelper.is_node_topology_prunable(n, float_graph, fw_info):
#                 prunable_nodes.append(n)
#
#         return prunable_nodes
#
#     # @staticmethod
#     def is_node_topology_prunable(node: BaseNode,
#                                   float_graph,
#                                   fw_info):
#
#         if fw_info.is_kernel_op(node.type):
#             next_node = node
#             while len(float_graph.out_edges(next_node)) == 1 and len(float_graph.in_edges(next_node)) == 1:
#                 next_node = float_graph.out_edges(next_node)[0].sink_node
#                 if fw_info.is_kernel_op(next_node.type):
#                     return True
#         return False
#
#     # @staticmethod
#     def get_pruning_sections(graph, fw_info) -> List[PruningSection]:
#         prunable_nodes = GraphHelper.get_prunable_nodes(graph, fw_info)
#         pruning_sections = []
#
#         # Iterate over each prunable node and find its corresponding section
#         for prunable_node in prunable_nodes:
#             conv_nodes, intermediate_nodes = get_section_nodes(prunable_node, graph, fw_info)
#             pruning_sections.append(PruningSection(conv_nodes, intermediate_nodes))
#
#         return pruning_sections
#
#
#
# def get_section_nodes(start_node: BaseNode,
#                       graph,
#                       fw_info) -> Tuple[List[BaseNode], List[BaseNode]]:
#
#     intermediate_nodes = []
#
#     # Follow the graph from the start_node to find the section's end
#     next_node = graph.out_edges(start_node)[0].sink_node
#     while not fw_info.is_kernel_op(next_node.type):
#         intermediate_nodes.append(next_node)
#         # Move to the next node in the section
#         next_node = graph.out_edges(next_node)[0].sink_node
#
#     assert fw_info.is_kernel_op(next_node.type)
#     conv_nodes = [start_node, next_node]
#
#     return conv_nodes, intermediate_nodes
#
#
