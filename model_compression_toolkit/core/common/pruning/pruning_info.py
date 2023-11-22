from typing import Dict, List
import numpy as np

class PruningInfo:
    """
    Class to store metadata about a pruned model, including pruning statistics,
    masks, importance scores, and parameter counts.
    """
    def __init__(self, original_params: int):
        """
        Initializes the PruningInfo with the original number of parameters.

        :param original_params: The total number of parameters in the unpruned model.
        """
        self.original_params = original_params
        self.compressed_params = None
        self.pruning_masks = {}  # Dictionary to store pruning masks for each layer
        self.importance_scores = {}  # Dictionary to store importance scores for each layer
        self.pruned_model_stats = {}  # Dictionary to store various statistics of the pruned model

    def add_pruning_mask(self, layer_name: str, mask: np.ndarray):
        """
        Adds a pruning mask for a specific layer.

        :param layer_name: The name of the layer.
        :param mask: The pruning mask for that layer as a numpy array.
        """
        self.pruning_masks[layer_name] = mask

    def add_importance_score(self, layer_name: str, score: np.ndarray):
        """
        Adds an importance score for a specific layer.

        :param layer_name: The name of the layer.
        :param score: The importance score for that layer as a numpy array.
        """
        self.importance_scores[layer_name] = score

    def set_compressed_params(self, compressed_params: int):
        """
        Sets the total number of parameters in the compressed (pruned) model.

        :param compressed_params: The total number of parameters in the pruned model.
        """
        self.compressed_params = compressed_params

    def add_pruned_model_stat(self, stat_name: str, value):
        """
        Adds a statistic about the pruned model.

        :param stat_name: The name of the statistic.
        :param value: The value of the statistic.
        """
        self.pruned_model_stats[stat_name] = value

    def get_pruning_mask(self, layer_name: str) -> np.ndarray:
        """
        Retrieves the pruning mask for a specific layer.

        :param layer_name: The name of the layer.
        :return: The pruning mask for that layer.
        """
        return self.pruning_masks.get(layer_name)

    def get_importance_score(self, layer_name: str) -> np.ndarray:
        """
        Retrieves the importance score for a specific layer.

        :param layer_name: The name of the layer.
        :return: The importance score for that layer.
        """
        return self.importance_scores.get(layer_name)

    def calculate_compression_ratio(self) -> float:
        """
        Calculates the compression ratio of the model after pruning.

        :return: The compression ratio.
        """
        if self.compressed_params is None:
            raise ValueError("Compressed model parameters not set.")
        return self.original_params / self.compressed_params

    def __str__(self):
        """
        Returns a string representation of the pruning information.

        :return: A string representation of the PruningInfo object.
        """
        info = f"Original Params: {self.original_params}\n"
        info += f"Compressed Params: {self.compressed_params}\n"
        info += f"Compression Ratio: {self.calculate_compression_ratio()}\n"
        info += f"Pruning Masks: {self.pruning_masks.keys()}\n"
        info += f"Importance Scores: {self.importance_scores.keys()}\n"
        info += f"Pruned Model Stats: {self.pruned_model_stats}\n"
        return info

# Example usage:
# pruning_info = PruningInfo(original_params=100000)
# pruning_info.add_pruning_mask('conv1', np.array([1, 0, 1, 1]))
# pruning_info.add_importance_score('conv1', np.array([0.1, 0.2, 0.3, 0.4]))
# pruning_info.set_compressed_params(95000)
# pruning_info.add_pruned_model_stat('memory_footprint', 1024)
#
# print(pruning_info)
