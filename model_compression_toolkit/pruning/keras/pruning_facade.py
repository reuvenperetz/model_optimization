# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Callable, Tuple

from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.constants import TENSORFLOW, FOUND_TF
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.pruning.pruner import Pruner
from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig
from model_compression_toolkit.core.common.pruning.pruning_info import PruningInfo
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.graph_prep_runner import read_model_to_graph
from model_compression_toolkit.core.keras.back2framework.float_model_builder import FloatKerasModelBuilder
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework import \
    TargetPlatformCapabilities

if FOUND_TF:
    from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
    from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
    from model_compression_toolkit.core.keras.keras_model_validation import KerasModelValidation
    from tensorflow.keras.models import Model
    from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL

    from model_compression_toolkit.core.common.quantization.quantization_config import DEFAULTCONFIG

    DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


    def keras_pruning_experimental(model: Model,
                                   target_kpi: KPI,
                                   representative_data_gen: Callable,
                                   pruning_config: PruningConfig = PruningConfig(),
                                   target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_KERAS_TPC) -> Tuple[Model, PruningInfo]:
        """



         """

        fw_impl = KerasImplementation()
        float_graph = read_model_to_graph(model,
                                          representative_data_gen,
                                          target_platform_capabilities,
                                          DEFAULT_KERAS_INFO,
                                          fw_impl)

        float_graph_with_compression_config = set_quantization_configuration_to_graph(float_graph,
                                                                                      quant_config=DEFAULTCONFIG,
                                                                                      mixed_precision_enable=False)

        pruner = Pruner(float_graph_with_compression_config,
                        DEFAULT_KERAS_INFO,
                        fw_impl,
                        target_kpi,
                        representative_data_gen,
                        pruning_config,
                        target_platform_capabilities)

        pruned_graph = pruner.get_pruned_graph()
        # pruning_info = pruner.get_pruning_info()

        # Build and return a trainable model in NN framework.
        pruned_model, _ = FloatKerasModelBuilder(graph=pruned_graph).build_model()

        pruning_info = None
        return pruned_model, pruning_info



else:
    # If tensorflow is not installed,
    # we raise an exception when trying to use these functions.
    def keras_pruning_experimental(*args, **kwargs):
        Logger.critical('Installing tensorflow is mandatory '
                        'when using keras_pruning_experimental. '
                        'Could not find Tensorflow package.')  # pragma: no cover
