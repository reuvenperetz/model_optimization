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


import itertools
import pprint

from model_compression_toolkit.common.logger import Logger
from model_compression_toolkit.common.hardware_representation.hardware2framework.operations_to_layers import \
    OperationsToLayers, OperationsSetToLayers
from model_compression_toolkit.common.hardware_representation.hardware2framework.framework_hardware_model_component import FrameworkHardwareModelComponent
from model_compression_toolkit.common.hardware_representation.hardware2framework.layer_filter_params import LayerFilterParams
from model_compression_toolkit.common.immutable import ImmutableClass
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.common.hardware_representation.op_quantization_config import QuantizationConfigOptions
from model_compression_toolkit.common.hardware_representation.operators import OperatorsSet, OperatorsSetBase
from model_compression_toolkit.common.hardware_representation.hardware_model import HardwareModel
from model_compression_toolkit.common.hardware_representation.hardware2framework.current_framework_hardware_model import _current_framework_hardware_model


class FrameworkHardwareModel(ImmutableClass):
    def __init__(self,
                 hw_model: HardwareModel,
                 name: str="base"):

        super().__init__()
        self.name = name
        assert isinstance(hw_model, HardwareModel)
        self.hw_model = hw_model
        self.op_sets_to_layers = OperationsToLayers()
        self.layer2qco, self.filterlayer2qco = {}, {}
        self.__hwm_opsets_not_used = [s.name for s in hw_model.operator_set]
        self.remove_fusing_names_from_not_used_list()

    def get_layers_by_opset_name(self, opset_name: str):
        opset = self.hw_model.get_opset_by_name(opset_name)
        if opset is None:
            Logger.warning(f'{opset_name} was not found in FrameworkHardwareModel.')
            return None
        return self.get_layers_by_opset(opset)

    def get_layers_by_opset(self, op: OperatorsSetBase):
        return self.op_sets_to_layers.get_layers_by_op(op)

    def get_fusing_patterns(self):
        res = []
        for p in self.hw_model.fusing_patterns:
            ops = [self.get_layers_by_opset(x) for x in p.operator_groups_list]
            res.extend(itertools.product(*ops))
        return [list(x) for x in res]


    def get_info(self):
        return {"Framework Hardware Model": self.name,
                "Hardware model": self.hw_model.get_info(),
                "Operations to layers": {op2layer.name:[l.__name__ for l in op2layer.layers] for op2layer in self.op_sets_to_layers.op_sets_to_layers}}

    def show(self):
        pprint.pprint(self.get_info(), sort_dicts=False, width=110)




    def append_component(self, hm_component: FrameworkHardwareModelComponent):
        if isinstance(hm_component, OperationsSetToLayers):
            self.op_sets_to_layers += hm_component
        else:
            raise Exception(f'Trying to append an unfamiliar HardwareModelComponent of type: {type(hm_component)}')

    def __enter__(self):
        _current_framework_hardware_model.set(self)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_value is not None:
            print(exc_value, exc_value.args)
            raise exc_value
        self.raise_warnings()
        self.layer2qco, self.filterlayer2qco = self._get_config_options_mapping()
        _current_framework_hardware_model.reset()
        self.initialized_done()
        return self

    def get_default_qc_options(self) -> QuantizationConfigOptions:
        return self.hw_model.default_qco

    def get_qco_by_node(self, node: BaseNode) -> QuantizationConfigOptions:
        if node is None:
            raise Exception(f'Can not retrieve QC options for None node')
        for fl, qco in self.filterlayer2qco.items():
            if fl.match(node):
                return qco
        if node.type in self.layer2qco:
            return self.layer2qco.get(node.type)
        return self.hw_model.default_qco

    def _get_config_options_mapping(self):
        layer2qco = {}
        filterlayer2qco = {}
        for op2layers in self.op_sets_to_layers.op_sets_to_layers:
            for l in op2layers.layers:
                qco = self.hw_model.get_config_options_by_operators_set(op2layers.name)
                if qco is None:
                    qco = self.hw_model.default_qco
                if isinstance(l, LayerFilterParams):
                    filterlayer2qco.update({l: qco})
                else:
                    layer2qco.update({l: qco})
        return layer2qco, filterlayer2qco


    def remove_fusing_names_from_not_used_list(self):
        for f in self.hw_model.fusing_patterns:
            for s in f.operator_groups_list:
                self.remove_opset_from_not_used_list(s.name)

    def remove_opset_from_not_used_list(self, opset_to_remove: str):
        if opset_to_remove in self.__hwm_opsets_not_used:
            self.__hwm_opsets_not_used.remove(opset_to_remove)

    def raise_warnings(self):
        for op in self.__hwm_opsets_not_used:
            Logger.warning(f'{op} is defined in hardware model, but is not used in framework hardware model.')


