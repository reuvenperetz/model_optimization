import numpy as np
from typing import Callable

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.hessian import HessianInfoService, HessianMode, HessianInfoGranularity
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.pruning.greedy_mask_calculator import GreedyMaskCalculator
from model_compression_toolkit.core.common.pruning.prunable_nodes import get_prunable_nodes
from model_compression_toolkit.core.common.pruning.prune_graph import build_pruned_graph
from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig, ChannelsFilteringStrategy, \
    ImportanceMetric
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities
import time

class Pruner:
    def __init__(self,
                 float_graph: Graph,
                 fw_info: FrameworkInfo,
                 fw_impl: FrameworkImplementation,
                 target_kpi: KPI,
                 representative_data_gen: Callable,
                 pruning_config: PruningConfig,
                 target_platform_capabilities: TargetPlatformCapabilities):

        self.float_graph=float_graph
        self.fw_info=fw_info
        self.fw_impl=fw_impl
        self.target_kpi=target_kpi
        self.representative_data_gen=representative_data_gen
        self.pruning_config=pruning_config
        self.target_platform_capabilities=target_platform_capabilities

        # TODO: BN are not folded so in the meanwhile hessian info is
        #  initialized here in addition to core.
        self.hessian_info_service = HessianInfoService(graph=float_graph,
                                                       representative_dataset=representative_data_gen,
                                                       fw_impl=fw_impl)

        self.mask_per_prunable_node = []
        self.mask_by_simd_group_per_prunable_node = []

        # Get all nodes that will be pruned
        self.prunable_nodes = get_prunable_nodes(float_graph=float_graph,
                                                 fw_info=fw_info)

    def get_pruned_graph(self):

        mean_score_per_prunable_node = None
        if self.pruning_config.importance_metric == ImportanceMetric.LFH:
            # Step 3: Compute LFH score for each channel that may be pruned.
            scores_per_prunable_node = self.hessian_info_service.fetch_scores_for_multiple_nodes(mode=HessianMode.WEIGHTS,
                                                                                                 granularity=HessianInfoGranularity.PER_OUTPUT_CHANNEL,
                                                                                                 nodes=self.prunable_nodes,
                                                                                                 required_size=self.pruning_config.num_score_approximations)
            # mean_score_per_prunable_node = [np.mean(node_scores, axis=0) for node_scores in scores_per_prunable_node]
            mean_score_per_prunable_node = {k:np.mean(v, axis=0) for k,v in zip(self.prunable_nodes, scores_per_prunable_node)}

        elif self.pruning_config.importance_metric == ImportanceMetric.RANDOM:
            tmp=[]
            for n in self.prunable_nodes:
                axis = self.fw_info.kernel_channels_mapping.get(n.type)[0]
                num_scores = n.get_weights_by_keys('kernel').shape[axis]
                tmp.append(np.random.random(num_scores))

            mean_score_per_prunable_node = {}
            for k,v in zip(self.prunable_nodes, tmp):
                mean_score_per_prunable_node[k]=v

        else:
            Logger.error(f"Not supported importance metric: {self.pruning_config.importance_metric}")


        # Step 4: Compute the mask by the remaining memory left and the
        # highest LFH of output channels simd-groups.
        assert mean_score_per_prunable_node is not None
        if self.pruning_config.channels_filtering_strategy == ChannelsFilteringStrategy.GREEDY:
            start = time.time()
            mask_calculator = GreedyMaskCalculator(self.prunable_nodes,
                                                   self.fw_info,
                                                   mean_score_per_prunable_node,
                                                   self.target_kpi,
                                                   self.float_graph,
                                                   self.fw_impl)

            self.mask_per_prunable_node = mask_calculator.get_greedy_channels_filtering_mask()
            end = time.time()
            print(f"Time for mask search: {end - start} seconds")
        else:
            Logger.error(f"Currently, ChannelsFilteringStrategy.GREEDY is supported only")

        # Step 5: Build a graph of the pruned model based on the float graph and pruning mask.
        pruned_graph = build_pruned_graph(self.float_graph,
                                          self.mask_per_prunable_node,
                                          self.fw_info,
                                          self.fw_impl)

        return pruned_graph


