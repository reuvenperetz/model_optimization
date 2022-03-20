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

from typing import Any, Dict

from model_compression_toolkit.common.graph.base_node import BaseNode


class LayerFilterParams:
    def __init__(self, layer: Any, *conditions, **kwargs):
        self.layer = layer
        self.conditions = conditions
        self.kwargs = kwargs
        self.__name__ = self.create_name()

    def __hash__(self):
        return hash(self.__name__)

    def __eq__(self, other):
        if not isinstance(other, LayerFilterParams):
            return False
        for self_c, other_c in zip(self.conditions, other.conditions):
            if self_c!=other_c:
                return False
        for k, v in self.kwargs.items():
            if k not in other.kwargs:
                return False
            else:
                if other.kwargs.get(k) != v:
                    return False
        return True

    def create_name(self):
        params = [f'{k}={v}' for k,v in self.kwargs.items()]
        params.extend([str(c) for c in self.conditions])
        params_str = ', '.join(params)
        return f'{self.layer.__name__}({params_str})'

    def match(self, node: BaseNode):
        if self.layer != node.type:
            return False
        layer_config = node.framework_attr
        if hasattr(node, "op_call_kwargs"):
            layer_config.update(node.op_call_kwargs)
        for attr, value in self.kwargs.items():
            if layer_config.get(attr) != value:
                return False
        for c in self.conditions:
            if not c.match(layer_config):
                return False
        return True
