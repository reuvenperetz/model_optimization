from typing import Dict

import copy
import numpy as np

from model_compression_toolkit.core.common import BaseNode


def build_pruned_graph(graph,
                       masks:Dict[BaseNode, np.ndarray],
                       fw_info,
                       fw_impl):

    graph_to_prune = copy.deepcopy(graph)
    prunable_nodes = graph_to_prune.get_pruning_sections_input_nodes(fw_info=fw_info)

    for node in prunable_nodes:
        mask = [v for k,v in masks.items() if k.name==node.name]
        assert len(mask)==1
        mask = mask[0]
        if np.sum(mask) > 0:
            _prune_node(node,
                        mask,
                        graph_to_prune,
                        fw_info,
                        fw_impl)

    return graph_to_prune

def _prune_node(node_to_prune: BaseNode,
                mask: np.ndarray,
                float_graph,
                fw_info,
                fw_impl):
    dependent_nodes = _get_dependent_nodes(node_to_prune,
                                           float_graph,
                                           fw_info)

    fw_impl.prune_node(node_to_prune,
                       mask,
                       fw_info,
                       last_section_node=False)

    intermediate_nodes = dependent_nodes[:-1]
    for n in intermediate_nodes:
        fw_impl.prune_node(n,
                           mask,
                           fw_info,
                           last_section_node=False)

    fw_impl.prune_node(dependent_nodes[-1],
                       mask,
                       fw_info,
                       last_section_node=True)


def _get_dependent_nodes(node: BaseNode,
                         float_graph,
                         fw_info):
    res = []
    while len(float_graph.out_edges(node)) == 1:
        node = float_graph.out_edges(node)[0].sink_node
        res.append(node)
        if fw_info.is_kernel_op(node.type):
            return res
    return []
