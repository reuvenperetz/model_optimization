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

from typing import List, Any

from model_compression_toolkit.common.logger import Logger
from model_compression_toolkit.common.hardware_representation.hardware2framework.current_framework_hardware_model import  _current_framework_hardware_model
from model_compression_toolkit.common.hardware_representation.hardware2framework.framework_hardware_model_component import FrameworkHardwareModelComponent
from model_compression_toolkit.common.hardware_representation.operators import OperatorsSet, OperatorSetConcat, \
    OperatorsSetBase


class OperationsToLayers:
    def __init__(self, op_sets_to_layers=None):
        if op_sets_to_layers is None:  # no mapping was added yet
            op_sets_to_layers = []
        else:
            assert isinstance(op_sets_to_layers, list)
        self.op_sets_to_layers = op_sets_to_layers
        self.validate_op_sets()

    def get_layers_by_op(self, op: OperatorsSetBase):
        for o in self.op_sets_to_layers:
            if op.name == o.name:
                return o.layers
        if isinstance(op, OperatorSetConcat):
            layers = []
            for o in op.op_set_list:
                layers.extend(self.get_layers_by_op(o))
            return layers
        Logger.warning(f'{op.name} is not in model.')
        return None

    def __add__(self, op_set_to_layers):
        assert isinstance(op_set_to_layers, OperationsSetToLayers)
        new_ops2layers = OperationsToLayers(self.op_sets_to_layers + [op_set_to_layers])
        new_ops2layers.validate_op_sets()
        return new_ops2layers

    def validate_op_sets(self):
        existing_layers = {}
        existing_opset_names = []
        for ops2layers in self.op_sets_to_layers:
            assert isinstance(ops2layers,
                              OperationsSetToLayers), f'Operators set should be of type OperationsSetToLayers but it ' \
                                                      f'is of type {type(ops2layers)}'
            is_opset_in_model = _current_framework_hardware_model.get(
            ).hw_model.is_opset_in_model(
                ops2layers.name)
            assert is_opset_in_model, f'{ops2layers.name} is not defined in the hardware model that is associated with the framework hardware model.'
            assert not (ops2layers.name in existing_opset_names), f'OperationsSetToLayers names should be unique, but {ops2layers.name} appears to violate it.'
            existing_opset_names.append(ops2layers.name)
            for layer in ops2layers.layers:
                qco_by_opset_name = _current_framework_hardware_model.get(
                ).hw_model.get_config_options_by_operators_set(
                    ops2layers.name)
                if layer in existing_layers:
                    raise Exception(f'Found layer {layer.__name__} in more than one '
                                    f'OperatorsSet')
                else:
                    existing_layers.update({layer: qco_by_opset_name})



class OperationsSetToLayers(FrameworkHardwareModelComponent):
    def __init__(self, op_set_name: str, layers: List[Any]):
        self.layers = layers
        super(OperationsSetToLayers, self).__init__(name=op_set_name)
        _current_framework_hardware_model.get().remove_opset_from_not_used_list(op_set_name)

    def __repr__(self):
        return f'{self.name} -> {[x.__name__ for x in self.layers]}'
