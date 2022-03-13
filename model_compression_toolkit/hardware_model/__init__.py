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
from model_compression_toolkit.common.hardware_model.framework_hardware_model import FrameworkHardwareModel
from model_compression_toolkit.common.hardware_model.fusing import Fusing
from model_compression_toolkit.common.hardware_model.global_config import GlobalConfig
from model_compression_toolkit.hardware_model.hardware_model import \
    HardwareModel, get_default_quantization_config_options
from model_compression_toolkit.common.hardware_model.operations_to_layers import OperationsToLayers, OperationsSetToLayers, OperatorSetConcat, OperatorsSet
from model_compression_toolkit.common.hardware_model.quantization_config \
    import \
    OpQuantizationConfig, QuantizationConfigOptions, QuantizationMethod



