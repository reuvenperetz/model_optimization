#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
from functools import partial

from typing import Tuple, Any, Dict, Callable

from model_compression_toolkit.xquant.common.constants import MODEL_OUTPUT_KEY, SIMILARITY_OUTPUT, \
    SIMILARITY_INTERMEDIATE, SIMILARITY_FULLQUANT, SIMILARITY_DISABLED_WEIGHTS_QUANT, \
    SIMILARITY_DISABLED_ACTIVATIONS_QUANT
from model_compression_toolkit.xquant.common.dataset_utils import DatasetUtils
from model_compression_toolkit.xquant.common.model_analyzer import ModelAnalyzer
from model_compression_toolkit.xquant.common.model_folding_utils import ModelFoldingUtils
from model_compression_toolkit.xquant.common.selective_quantization import SelectiveQuantization
from model_compression_toolkit.xquant.common.similarity_functions import SimilarityFunctions
from model_compression_toolkit.logger import Logger

class SimilarityCalculator:
    """
    A class to calculate the similarity between two models (that are often referred as float
    and quantized models). It utilizes various utility classes for dataset preparation, model folding,
    similarity computation, and model analysis.
    """

    def __init__(self,
                 dataset_utils: DatasetUtils,
                 model_folding: ModelFoldingUtils,
                 similarity_functions: SimilarityFunctions,
                 model_analyzer_utils: ModelAnalyzer,
                 selective_quantization: SelectiveQuantization,
                 device: str = None):
        """
        Initialize the SimilarityCalculator with required utilities.

        Args:
            dataset_utils (DatasetUtils): Utility class for dataset preparation.
            model_folding (ModelFoldingUtils): Utility class for model folding operations.
            similarity_functions (SimilarityFunctions): Class containing similarity functions.
            model_analyzer_utils (ModelAnalyzer): Utility class for model analysis.
            selective_quantization:
            device (str, optional): Device to perform computations on (e.g., 'cpu', 'cuda'). Defaults to None.
        """
        self.dataset_utils = dataset_utils
        self.model_folding = model_folding
        self.similarity_functions = similarity_functions
        self.model_analyzer_utils = model_analyzer_utils
        self.selective_quantization = selective_quantization
        self.device = device

    @staticmethod
    def compute_tensors_similarity(tensors_to_compare: Tuple[Any, Any],
                                   similarity_metrics: Dict[str, Callable]) -> Dict[str, float]:
        """
        Compute the similarity between two tensors using provided similarity metrics.

        Args:
            tensors_to_compare (Tuple[Any, Any]): Tensors to compare by computing their similarity.
            similarity_metrics (Dict[str, Callable]): A dictionary with similarity metric names and functions.

        Returns:
            Dict[str, float]: A dictionary of similarity metric names and their computed values.
        """
        x, y = tensors_to_compare
        similarity_metrics = {k: v(x, y) for k, v in similarity_metrics.items()}
        return similarity_metrics

    def _get_float_to_quantized_compare_points(self,
                                               quantized_model: Any,
                                               float_model: Any) -> Dict[str, str]:
        """
        Map corresponding layers between the float and quantized models for comparison.

        Args:
            quantized_model (Any): The quantized model.
            float_model (Any): The float model.

        Returns:
            Dict[str, str]: A dictionary mapping float model layer names to quantized model layer names.
        """
        # Identify the points in the quantized model to compare.
        quant_points_names = self.model_analyzer_utils.identify_quantized_compare_points(quantized_model)

        float_name2quant_name = {}

        # Extract the names of the layers in the float model.
        float_layers_names = self.model_analyzer_utils.extract_float_layer_names(float_model)

        # Map each quantized layer to the corresponding float layer.
        for quant_point in quant_points_names:
            candidate_float_layer_name = self.model_analyzer_utils.find_corresponding_float_layer(
                quant_compare_point=quant_point, quantized_model=quantized_model)

            if candidate_float_layer_name in float_layers_names:
                if candidate_float_layer_name not in float_name2quant_name:
                    float_name2quant_name[candidate_float_layer_name] = quant_point
                else:
                    Logger.critical(f"Duplicate mapping found for layer: {candidate_float_layer_name}.")
            else:
                Logger.warning(
                    f"Could not find a matching layer in the float model for layer with name {quant_point}, "
                    f"skipping it in similarity metrics comparison points computation.")

        return float_name2quant_name

    def compute_similarity_metrics(self,
                                   float_model,
                                   quantized_model,
                                   dataset,
                                   custom_similarity_metrics=None,
                                   is_validation=False):

        dataset = self._prepare_dataset(dataset, is_validation)
        float_model = self._create_folded_float_model(float_model, dataset)

        metrics = self._get_metrics_to_compute(custom_similarity_metrics)
        layer_mapping = self._map_float_to_quantized_layers(float_model, quantized_model)

        similarity_results = self._initialize_similarity_results(metrics, layer_mapping)

        disabled_w_model, disabled_w_layer_mapping = self._disable_weights_quantization(float_model, quantized_model)
        disabled_a_model = self._disable_activations_quantization(float_model, quantized_model)

        for input_data in dataset():
            self._compute_similarities_for_input(
                input_data,
                float_model,
                quantized_model,
                disabled_w_model,
                disabled_a_model,
                layer_mapping,
                disabled_w_layer_mapping,
                metrics,
                similarity_results
            )

        return self._aggregate_similarities(similarity_results)

    def _prepare_dataset(self, dataset, is_validation):
        return partial(self.dataset_utils.prepare_dataset, dataset=dataset, is_validation=is_validation,
                       device=self.device)

    def _create_folded_float_model(self, float_model, dataset):
        return self.model_folding.create_float_folded_model(float_model, dataset)

    def _get_metrics_to_compute(self, custom_metrics):
        metrics = self.similarity_functions.get_default_similarity_metrics()
        if custom_metrics:
            if not isinstance(custom_metrics, dict):
                raise ValueError("custom_metrics should be a dictionary")
            metrics.update(custom_metrics)
        return metrics

    def _map_float_to_quantized_layers(self, float_model, quantized_model):
        return self._get_float_to_quantized_compare_points(quantized_model, float_model)

    def _initialize_similarity_results(self, metrics, layer_mapping):
        return {
            model_type: {
                'output': {metric: [] for metric in metrics},
                'intermediate': {layer: {metric: [] for metric in metrics} for layer in layer_mapping.values()}
            }
            for model_type in ['full_quant', 'disabled_weights', 'disabled_activations']
        }

    def _disable_weights_quantization(self, float_model, quantized_model):
        return self.selective_quantization.disable_weights_quantization(float_model, quantized_model)

    def _disable_activations_quantization(self, float_model, quantized_model):
        return self.selective_quantization.disable_activations_quantization(float_model, quantized_model)

    def _compute_similarities_for_input(self,
                                        input_data,
                                        float_model,
                                        quantized_model,
                                        disabled_w_model,
                                        disabled_a_model,
                                        layer_mapping,
                                        disabled_w_layer_mapping,
                                        metrics,
                                        results):
        float_activations = self._extract_activations(float_model, list(layer_mapping.keys()), input_data)
        quant_activations = self._extract_activations(quantized_model, list(layer_mapping.values()), input_data)
        disabled_w_activations = self._extract_activations(disabled_w_model, list(disabled_w_layer_mapping.values()), input_data)
        disabled_a_activations = self._extract_activations(disabled_a_model, list(layer_mapping.values()), input_data)

        self._compute_model_similarities(float_activations,
                                         quant_activations,
                                         layer_mapping,
                                         metrics,
                                         results['full_quant'])

        self._compute_model_similarities(float_activations,
                                         disabled_a_activations,
                                         layer_mapping,
                                         metrics,
                                         results['disabled_activations'])

        self._compute_model_similarities(float_activations,
                                         disabled_w_activations,
                                         layer_mapping,
                                         metrics,
                                         results['disabled_weights'])

    def _extract_activations(self, model, layer_names, input_data):
        return self.model_analyzer_utils.extract_model_activations(model, layer_names, input_data)

    def _compute_model_similarities(self,
                                    float_activations,
                                    model_activations,
                                    layer_mapping,
                                    metrics,
                                    results):
        self._compute_output_similarities(float_activations,
                                          model_activations,
                                          metrics,
                                          results['output'])
        self._compute_intermediate_similarities(float_activations,
                                                model_activations,
                                                layer_mapping,
                                                metrics,
                                                results['intermediate'])

    def _compute_output_similarities(self, float_activations, model_activations, metrics, results):
        similarities = self.compute_tensors_similarity(
            (float_activations[MODEL_OUTPUT_KEY], model_activations[MODEL_OUTPUT_KEY]), metrics)
        for metric, value in similarities.items():
            results[metric].append(value)

    def _compute_intermediate_similarities(self,
                                           float_activations,
                                           model_activations,
                                           layer_mapping,
                                           metrics,
                                           results):
        for float_layer, quant_layer in layer_mapping.items():
            similarities = self.compute_tensors_similarity(
                (float_activations[float_layer], model_activations[quant_layer]), metrics)
            for metric, value in similarities.items():
                results[quant_layer][metric].append(value)

    def _aggregate_similarities(self, results):
        aggregated_results = {}
        for model_type, model_results in results.items():
            aggregated_results[model_type] = {
                'output': self._average_metrics(model_results['output']),
                'intermediate': {layer: self._average_metrics(layer_metrics) for layer, layer_metrics in
                                 model_results['intermediate'].items()}
            }
        return aggregated_results

    def _average_metrics(self, metrics):
        return {metric: sum(values) / len(values) if values else 0 for metric, values in metrics.items()}

    def _merge_layer_mappings(self, layer_mapping, disabled_w_layer_mapping):
        merged_mapping = {}
        for quant_layer, float_layer in layer_mapping.items():
            if float_layer in disabled_w_layer_mapping:
                merged_mapping[quant_layer] = disabled_w_layer_mapping[float_layer]
            else:
                merged_mapping[quant_layer] = float_layer
        return merged_mapping

    # def compute_similarity_metrics(self,
    #                                float_model: Any,
    #                                quantized_model: Any,
    #                                dataset: Callable,
    #                                custom_similarity_metrics: Dict[str, Callable] = None,
    #                                is_validation: bool = False):
    #
    #     dataset = partial(self.dataset_utils.prepare_dataset,
    #                       dataset=dataset,
    #                       is_validation=is_validation,
    #                       device=self.device)
    #
    #     # Create a folded version of the float model.
    #     float_model = self.model_folding.create_float_folded_model(float_model=float_model,
    #                                                                representative_dataset=dataset)
    #
    #     # Gather similarity metrics to compute (default and custom).
    #     similarity_metrics_to_compute = self.similarity_functions.get_default_similarity_metrics()
    #     if custom_similarity_metrics:
    #         if not isinstance(custom_similarity_metrics, dict):
    #             Logger.critical(
    #                 f"custom_similarity_metrics should be a dictionary but is of type "
    #                 f"{type(custom_similarity_metrics)}.")
    #         similarity_metrics_to_compute.update(custom_similarity_metrics)
    #
    #     # Map float model layers to quantized model layers for comparison.
    #     float_name2quant_name = self._get_float_to_quantized_compare_points(float_model=float_model,
    #                                                                         quantized_model=quantized_model)
    #
    #     # Initialize dictionaries to store similarity metrics.
    #     output_similarity_metrics = {key: [] for key in similarity_metrics_to_compute.keys()}
    #     intermediate_similarity_metrics = {layer: {key: [] for key in similarity_metrics_to_compute.keys()} for layer in float_name2quant_name.values()}
    #
    #     disabled_w_output_similarity = {key: [] for key in similarity_metrics_to_compute.keys()}
    #     disabled_w_intermediate_similarity = {layer: {key: [] for key in similarity_metrics_to_compute.keys()} for layer in float_name2quant_name.values()}
    #
    #     disabled_a_output_similarity = {key: [] for key in similarity_metrics_to_compute.keys()}
    #     disabled_a_intermediate_similarity = {layer: {key: [] for key in similarity_metrics_to_compute.keys()} for layer in float_name2quant_name.values()}
    #
    #
    #
    #     disabled_w_model, wrapped_to_unwrapped_layers_names_mapping = self.selective_quantization.disable_weights_quantization(float_model=float_model,
    #                                                                                                       quantized_model=quantized_model)
    #     unwrapped_to_wrapped_layers_names_mapping = {}
    #     for w, un in wrapped_to_unwrapped_layers_names_mapping.items():
    #         unwrapped_to_wrapped_layers_names_mapping[un]=w
    #
    #     disable_w_layers_names = list(float_name2quant_name.values())
    #     for i, l_name in enumerate(disable_w_layers_names):
    #         if l_name in wrapped_to_unwrapped_layers_names_mapping:
    #             disable_w_layers_names[i] = wrapped_to_unwrapped_layers_names_mapping[l_name]
    #
    #     disabled_act_model = self.selective_quantization.disable_activations_quantization(float_model=float_model,
    #                                                                                       quantized_model=quantized_model)
    #
    #     # Iterate over the dataset and compute similarity metrics.
    #     for x in dataset():
    #         # Extract activations and predictions from both models.
    #         float_activations = self.model_analyzer_utils.extract_model_activations(float_model, list(float_name2quant_name.keys()), x)
    #         quant_activations = self.model_analyzer_utils.extract_model_activations(quantized_model, list(float_name2quant_name.values()), x)
    #
    #         disable_act_model_activations = self.model_analyzer_utils.extract_model_activations(disabled_act_model, list(float_name2quant_name.values()), x)
    #         disable_w_model_activations = self.model_analyzer_utils.extract_model_activations(disabled_w_model, disable_w_layers_names, x)
    #
    #
    #         new_disable_w_model_activations = {}
    #         for k,v in disable_w_model_activations.items():
    #             if k in unwrapped_to_wrapped_layers_names_mapping:
    #                 k2 = unwrapped_to_wrapped_layers_names_mapping[k]
    #                 new_disable_w_model_activations[k2]=v
    #             else:
    #                 new_disable_w_model_activations[k]=v
    #
    #         self.compute_similarity_on_current_inputs(float_activations,
    #                                                   float_name2quant_name,
    #                                                   intermediate_similarity_metrics,
    #                                                   output_similarity_metrics,
    #                                                   quant_activations,
    #                                                   similarity_metrics_to_compute)
    #
    #
    #         self.compute_similarity_on_current_inputs(float_activations,
    #                                                   float_name2quant_name,
    #                                                   disabled_w_intermediate_similarity,
    #                                                   disabled_w_output_similarity,
    #                                                   new_disable_w_model_activations,
    #                                                   similarity_metrics_to_compute)
    #
    #
    #         self.compute_similarity_on_current_inputs(float_activations,
    #                                                   float_name2quant_name,
    #                                                   disabled_a_intermediate_similarity,
    #                                                   disabled_a_output_similarity,
    #                                                   disable_act_model_activations,
    #                                                   similarity_metrics_to_compute)
    #
    #     _similarities = {}
    #
    #     aggregated_output_similarity_metrics, intermediate_similarity_metrics = self.aggregate_similarities_on_all_batches(intermediate_similarity_metrics, output_similarity_metrics)
    #
    #
    #     _similarities[SIMILARITY_FULLQUANT] = {SIMILARITY_OUTPUT: aggregated_output_similarity_metrics,
    #                                            SIMILARITY_INTERMEDIATE: intermediate_similarity_metrics}
    #
    #
    #     disabled_w_output_similarity, disabled_w_intermediate_similarity = self.aggregate_similarities_on_all_batches(disabled_w_intermediate_similarity, disabled_w_output_similarity)
    #
    #
    #     _similarities[SIMILARITY_DISABLED_WEIGHTS_QUANT] = {SIMILARITY_OUTPUT: disabled_w_output_similarity,
    #                                                         SIMILARITY_INTERMEDIATE: disabled_w_intermediate_similarity}
    #
    #
    #
    #     disabled_a_output_similarity, disabled_a_intermediate_similarity  = self.aggregate_similarities_on_all_batches(disabled_a_intermediate_similarity, disabled_a_output_similarity)
    #
    #     _similarities[SIMILARITY_DISABLED_ACTIVATIONS_QUANT] = {SIMILARITY_OUTPUT: disabled_a_output_similarity,
    #                                                             SIMILARITY_INTERMEDIATE: disabled_a_intermediate_similarity}
    #
    #     return _similarities
    #
    # def aggregate_similarities_on_all_batches(self,
    #                                           intermediate_similarity_metrics,
    #                                           output_similarity_metrics):
    #
    #     # Aggregate the output similarity metrics.
    #     aggregated_output_similarity_metrics = {key: sum(value) / len(value) for key, value in
    #                                             output_similarity_metrics.items()}
    #     # Aggregate the intermediate similarity metrics for each layer.
    #     for layer_name, layer_similarity_metrics in intermediate_similarity_metrics.items():
    #         for similarity_name, similarity_values_list in layer_similarity_metrics.items():
    #             if len(similarity_values_list) == 0:
    #                 Logger.critical(f"Can not average similarities of an empty list.")
    #             intermediate_similarity_metrics[layer_name][similarity_name] = sum(similarity_values_list) / len(similarity_values_list)
    #
    #     return aggregated_output_similarity_metrics, intermediate_similarity_metrics
    #
    # def compute_similarity_on_current_inputs(self,
    #                                          float_activations,
    #                                          float_name2quant_name,
    #                                          intermediate_similarity_metrics,
    #                                          output_similarity_metrics,
    #                                          quant_activations,
    #                                          similarity_metrics_to_compute):
    #
    #     float_predictions = float_activations[MODEL_OUTPUT_KEY]
    #     quant_predictions = quant_activations[MODEL_OUTPUT_KEY]
    #     # Compute similarity metrics for the output predictions.
    #     output_results = self.compute_tensors_similarity((float_predictions, quant_predictions),
    #                                                      similarity_metrics_to_compute)
    #     for key in output_similarity_metrics:
    #         output_similarity_metrics[key].append(output_results[key])
    #     # Compute similarity metrics for each intermediate layer.
    #     for float_layer, quant_layer in float_name2quant_name.items():
    #         intermediate_results = self.compute_tensors_similarity(
    #             (float_activations[float_layer], quant_activations[quant_layer]),
    #             similarity_metrics_to_compute)
    #         for key in intermediate_similarity_metrics[quant_layer]:
    #             intermediate_similarity_metrics[quant_layer][key].append(intermediate_results[key])
