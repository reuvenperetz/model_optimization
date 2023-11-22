import numpy as np
from typing import List, Dict

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.pruning.memory_helper import MemoryHelper
# from model_compression_toolkit.core.common.pruning.memory_utils import get_pruned_graph_memory
from model_compression_toolkit.logger import Logger


class GreedyMaskCalculator:
    def __init__(self,
                 prunable_nodes: List[BaseNode],
                 fw_info: FrameworkInfo,
                 score_by_node: Dict[BaseNode, np.ndarray],
                 target_kpi: KPI,
                 graph,
                 fw_impl):

        self.prunable_nodes = prunable_nodes
        self.fw_info = fw_info
        self.score_by_node = score_by_node
        self.target_kpi = target_kpi
        self.graph = graph
        self.fw_impl = fw_impl

        self.mask_simd = {}
        self.simd_groups_indices = {}
        self.simd_groups_scores = {}
        self.mask = {}

    def update_simd_mask(self,
                         node: BaseNode,
                         group_index: int,
                         value: int):
        assert value in [0,1]
        self.mask_simd[node][group_index] = value
        node_mask_indices = self.simd_groups_indices[node][group_index]
        self.mask[node][node_mask_indices] = value


    def get_greedy_channels_filtering_mask(self):

        # init mask for each layer where each layer has at least one group of
        # SIMD output-channels.
        for prunable_node in self.prunable_nodes:
            oc_axis = self.fw_info.kernel_channels_mapping.get(prunable_node.type)[0]
            kernel_attr = self.fw_info.get_kernel_op_attributes(prunable_node.type)[0]
            kernel_shape = list(prunable_node.get_weights_by_keys(kernel_attr).shape)
            num_oc = kernel_shape[oc_axis]
            layer_mask = np.zeros(num_oc)
            layer_num_simd_groups = int(max(np.ceil(num_oc / prunable_node.get_simd()), 1))  # at least one group of channels will remain in the layer
            layer_mask_per_simd_group = np.zeros(layer_num_simd_groups)

            self.mask_simd[prunable_node] = layer_mask_per_simd_group
            self.mask[prunable_node] = layer_mask

        for prunable_node, node_scores in self.score_by_node.items():
            self.simd_groups_scores[prunable_node], self.simd_groups_indices[prunable_node] = self.sort_and_group(
                node_scores,
                prunable_node.get_simd())

            self.update_simd_mask(node=prunable_node,
                                  group_index=0,
                                  value=1)

        current_memory = MemoryHelper.get_pruned_graph_memory(graph=self.graph,
                                                              fw_info=self.fw_info,
                                                              masks=self.mask)

        if current_memory > self.target_kpi.weights_memory:
            Logger.error(f"Minimal required memory is {current_memory} but target KPI"
                         f" is {self.target_kpi.weights_memory}")

        while current_memory < self.target_kpi.weights_memory and self.is_there_pruned_channel():
            print(f"Current memory: {current_memory}")
            node_to_remain, group_to_remain_idx = self._get_best_simd_group_candidate()
            self.update_simd_mask(node=node_to_remain,
                                  group_index=group_to_remain_idx,
                                  value=1)
            current_memory = MemoryHelper.get_pruned_graph_memory(self.graph,
                                                                  self.mask,
                                                                  self.fw_info)

        if current_memory > self.target_kpi.weights_memory:
            self.update_simd_mask(node=node_to_remain,
                                  group_index=group_to_remain_idx,
                                  value=0)

        return self.mask

    def sort_and_group(self, array, simd):
        # Get the indices that would sort the array in descending order
        sorted_indices = np.argsort(array)[::-1]
        sorted_array = array[sorted_indices]
        # Calculate the number of groups
        num_groups = len(sorted_array) // simd + (1 if len(sorted_array) % simd else 0)
        # Split the indices and values into groups
        indices_groups = [sorted_indices[i * simd:(i + 1) * simd] for i in range(num_groups)]
        scores_groups = [sorted_array[i * simd:(i + 1) * simd] for i in range(num_groups)]
        return scores_groups, indices_groups


    def _get_best_simd_group_candidate(self):
        best_scores = {}
        best_indices = {}
        for node, mask in self.mask_simd.items():
            # Get the index of the first occurrence of zero
            index_of_first_zero = int(np.argmax(mask == 0))
            if index_of_first_zero==0: # All ones
                best_scores[node]=-np.inf
            else:
                best_scores[node] = np.sum(self.simd_groups_scores[node][index_of_first_zero])
            best_indices[node] = index_of_first_zero

        node = max(best_scores, key=best_scores.get)
        idx = best_indices[node]
        return node, idx

    def is_there_pruned_channel(self):
        for m in list(self.mask.values()):
            if 0 in m:
                return True
        return False