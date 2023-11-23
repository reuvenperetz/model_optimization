from enum import Enum

from model_compression_toolkit.constants import PRUNING_NUM_SCORE_APPROXIMATIONS
from model_compression_toolkit.logger import Logger


class ImportanceMetric(Enum):
    LFH = 0  # Hessian approximation based on weights
    RANDOM = 2


class ChannelsFilteringStrategy(Enum):
    GREEDY = 0


class PruningConfig:
    def __init__(self,
                 num_score_approximations: int = PRUNING_NUM_SCORE_APPROXIMATIONS,
                 importance_metric: ImportanceMetric = ImportanceMetric.LFH,
                 channels_filtering_strategy: ChannelsFilteringStrategy = ChannelsFilteringStrategy.GREEDY):

        self.num_score_approximations = num_score_approximations
        self.importance_metric = importance_metric
        self.channels_filtering_strategy = channels_filtering_strategy
