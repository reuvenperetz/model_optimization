from typing import Dict

import copy
import numpy as np

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.pruning.pruning_section import PruningSectionMask


def build_pruned_graph(graph: Graph,
                       masks:Dict[BaseNode, np.ndarray],
                       fw_info: FrameworkInfo,
                       fw_impl: FrameworkImplementation):

    graph_to_prune = copy.deepcopy(graph)
    prunable_nodes = graph_to_prune.get_pruning_sections_input_nodes(fw_info=fw_info, fw_impl=fw_impl)
    pruning_sections = graph_to_prune.get_pruning_sections(fw_info=fw_info, fw_impl=fw_impl)
    assert len(pruning_sections)==len(prunable_nodes)
    for input_pruning_section_node, pruning_section in zip(prunable_nodes, pruning_sections):
        mask = [v for k,v in masks.items() if k.name==input_pruning_section_node.name]
        assert len(mask)==1
        mask = mask[0]
        if np.sum(mask) > 0:
            section_mask = PruningSectionMask(input_node_ic_mask=None,
                                              input_node_oc_mask=mask,
                                              output_node_oc_mask=None)
            pruning_section.apply_inner_section_mask(section_mask, fw_impl, fw_info)
    return graph_to_prune
