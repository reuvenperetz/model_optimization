# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.keras.back2framework.float_model_builder import FloatKerasModelBuilder
from model_compression_toolkit.core.keras.back2framework.fully_quantized_model_builder import \
    FullyQuantizedKerasModelBuilder
from model_compression_toolkit.core.keras.back2framework.mixed_precision_model_builder import \
    MixedPrecisionKerasModelBuilder
from model_compression_toolkit.core.keras.back2framework.quantized_model_builder import QuantizedKerasModelBuilder

keras_model_builders = {ModelBuilderMode.QUANTIZED: QuantizedKerasModelBuilder,
                        ModelBuilderMode.FLOAT: FloatKerasModelBuilder,
                        ModelBuilderMode.MIXEDPRECISION: MixedPrecisionKerasModelBuilder,
                        ModelBuilderMode.FULLY_QUANTIZED: FullyQuantizedKerasModelBuilder}


def get_keras_model_builder(mode: ModelBuilderMode) -> type:
    """
    Return a Keras model builder given a ModelBuilderMode.

    Args:
        mode: Mode of the Keras model builder.

    Returns:
        Keras model builder for the given mode.
    """

    if not isinstance(mode, ModelBuilderMode):
        Logger.error(f'get_keras_model_builder expects a mode of type ModelBuilderMode, but {type(mode)} was passed.')
    if mode is None:
        Logger.error(f'get_keras_model_builder received a mode which is None')
    if mode not in keras_model_builders.keys():
        Logger.error(f'mode {mode} is not in keras model builders factory')
    return keras_model_builders.get(mode)
