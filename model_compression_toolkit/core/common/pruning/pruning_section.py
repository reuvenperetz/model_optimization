from typing import List

import numpy as np

from model_compression_toolkit.core.common.graph.base_node import BaseNode


class PruningSectionMask:
    def __init__(self,
                 input_node_ic_mask: np.ndarray,
                 input_node_oc_mask: np.ndarray,
                 output_node_oc_mask: np.ndarray
                 ):
        self.input_node_ic_mask = input_node_ic_mask
        self.input_node_oc_mask = input_node_oc_mask
        self.output_node_oc_mask = output_node_oc_mask


class PruningSection:

    def __init__(self,
                 input_node:BaseNode,
                 intermediate_nodes: List[BaseNode],
                 output_node: BaseNode):
        self.input_node = input_node
        self.intermediate_nodes = intermediate_nodes
        self.output_node = output_node

