import copy
from typing import Any, List

from model_compression_toolkit.common import Graph, BaseNode
from model_compression_toolkit.common.hardware_model.layer_filter_params import \
    LayerFilterParams
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation


def fuse_graph(g: Graph, fw_impl: FrameworkImplementation):
    graph = copy.deepcopy(g)
    fw_hw_model = graph.fw_hw_model
    patterns = fw_hw_model.get_fusing_patterns()
    # node_sort = list(topological_sort(graph))  # hold nodes after sorting them
    for p in patterns:
        for node in graph.nodes:
            nodes = is_pattern_match(p, node, graph, fw_impl)
            for n in nodes[:-1]:
                n.activation_quantization_cfg.enable_activation_quantization = False
    return graph




def is_pattern_match(p:List[Any], node:BaseNode, graph:Graph, fw_impl: FrameworkImplementation):
    res = []
    if len(p)==0:
        return []
    if node.type == p[0] or (isinstance(p[0], LayerFilterParams) and p[0].match(fw_impl.node_builder(node))):
        res.append(node)
        for next_n in graph.get_next_nodes(node):
            x=is_pattern_match(p[1:], next_n, graph, fw_impl)
            if len(x)==len(p)-1:
                res.extend(x)
                return res
        # if len(graph.get_next_nodes(node))==0:
        #     return len(p)==1
    return res




# def is_node_match(e,node):
#     if isinstance(e, LayerFilterParams):
#         e.match(node.type(**node.framework_attr))


