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
from typing import Dict, List, Tuple

import torch
from mct_quantizers.pytorch.quantize_wrapper import PytorchQuantizationWrapper
from model_compression_toolkit.xquant.common.constants import MODEL_OUTPUT_KEY

from model_compression_toolkit.xquant.common.model_analyzer import ModelAnalyzer


class PytorchModelAnalyzer(ModelAnalyzer):
    """
    This class provides utilities for analyzing Pytorch models, specifically for
    extracting activations and comparing float and quantized models.
    """

    def extract_model_activations(self,
                                  model: torch.nn.Module,
                                  layers_names: List[str],
                                  data: List[torch.Tensor]) -> Dict[str, torch.Tensor]:

        def _compute_activations(name: str, activations: dict):
            """
            Creates a hook function to capture the activations of a layer.

            Args:
                name (str): The name of the layer.
                activations (dict): The dictionary to store the activations.

            Returns:
                hook (function): The hook function to register with the layer.
            """
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        _activations = {}

        # Register hooks for all layers in the model
        for layer_name in layers_names:
            layer = dict([*model.named_modules()])[layer_name]
            layer.register_forward_hook(_compute_activations(layer_name, _activations))

        # Perform a forward pass with the input data and capture activations
        with torch.no_grad():
            _predictions = model(*data)

        _activations[MODEL_OUTPUT_KEY] = _predictions
        return _activations

    def identify_quantized_compare_points(self,
                                          quantized_model: torch.nn.Module) -> List[str]:
        """
        Identifies points in the quantized model to compare with the float model.

        Args:
            quantized_model (torch.nn.Module): The quantized model.

        Returns:
            List[str]: A list of layer names in the quantized model to compare.
        """
        return [n for n, m in quantized_model.named_modules() if isinstance(m, PytorchQuantizationWrapper)]

    def find_corresponding_float_layer(self,
                                       quant_compare_point: str,
                                       quantized_model: torch.nn.Module) -> str:
        """
        Finds the corresponding float model layer for a given quantized model layer.
        In pytorch, we assume the name is the same in the float model, thus we return quant_compare_point.

        Args:
            quant_compare_point (str): The name of the layer in the quantized model.
            quantized_model (torch.nn.Module): The quantized model.

        Returns:
            str: The name of the corresponding layer in the float model.
        """
        return quant_compare_point

    def extract_float_layer_names(self,
                                  float_model: torch.nn.Module) -> List[str]:
        """
        Extracts the names of all layers in the float model.

        Args:
            float_model (torch.nn.Module): The float model.

        Returns:
            List[str]: A list of layer names in the float model.
        """
        return [n for n, m in float_model.named_modules()]

