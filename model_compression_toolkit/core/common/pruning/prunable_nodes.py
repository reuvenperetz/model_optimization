from networkx.algorithms.dag import topological_sort
from typing import List

from model_compression_toolkit.core.common import BaseNode


# # Get all nodes that will be pruned
# def get_prunable_nodes(float_graph, fw_info) -> List[BaseNode]:
#     prunable_nodes = []
#     for n in list(topological_sort(float_graph)):
#         if is_node_topology_prunable(n, float_graph, fw_info):
#             prunable_nodes.append(n)
#
#     return prunable_nodes
#
#
# def is_node_topology_prunable(node: BaseNode,
#                               float_graph,
#                               fw_info):
#
#     if fw_info.is_kernel_op(node.type):
#         next_node = node
#         while len(float_graph.out_edges(next_node)) == 1 and len(float_graph.in_edges(next_node)) == 1:
#             next_node = float_graph.out_edges(next_node)[0].sink_node
#             if fw_info.is_kernel_op(next_node.type):
#                 return True
#     return False
