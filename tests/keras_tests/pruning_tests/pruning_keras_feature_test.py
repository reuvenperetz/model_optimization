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
import model_compression_toolkit as mct
from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest


class PruningKerasFeatureTest(BaseKerasFeatureNetworkTest):
    def __init__(self,
                 unit_test,
                 num_calibration_iter=1,
                 val_batch_size=1,
                 num_of_inputs=1,
                 input_shape=(8, 8, 3)):

        super().__init__(unit_test=unit_test,
                         val_batch_size=val_batch_size,
                         num_calibration_iter=num_calibration_iter,
                         num_of_inputs=num_of_inputs,
                         input_shape=input_shape)

    def get_pruning_config(self):
        return PruningConfig()

    def run_test(self):
        feature_networks = self.create_networks()
        feature_networks = feature_networks if isinstance(feature_networks, list) else [feature_networks]
        for model_float in feature_networks:
            pruned_model, pruning_info = mct.pruning.keras_pruning_experimental(model=model_float,
                                                                                target_kpi=self.get_kpi(),
                                                                                representative_data_gen=self.representative_data_gen_experimental,
                                                                                pruning_config=self.get_pruning_config(),
                                                                                target_platform_capabilities=self.get_tpc())

            self.compare(pruned_model,
                         model_float,
                         input_x=self.representative_data_gen(),
                         quantization_info=pruning_info)

