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

import operator
from typing import Any, Callable, Dict


class Filter:
    def match(self, layer_config: Dict[str, Any]):
        raise Exception('Filter did not implement match')


class OrAttributeFilter(Filter):
    def __init__(self, *filters):
        self.filters = filters

    def match(self, layer_config: Dict[str, Any]):
        for f in self.filters:
            if f.match(layer_config):
                return True
        return False

    def __repr__(self):
        return ' | '.join([str(f) for f in self.filters])


class AndAttributeFilter(Filter):
    def __init__(self, *filters):
        self.filters = filters

    def match(self, layer_config: Dict[str, Any]):
        for f in self.filters:
            if not f.match(layer_config):
                return False
        return True

    def __repr__(self):
        return ' & '.join([str(f) for f in self.filters])


class AttributeFilter(Filter):
    def __init__(self, attr: str, value: Any, op: Callable):
        self.attr = attr
        self.value = value
        self.op = op

    def __eq__(self, other):
        if not isinstance(other, AttributeFilter):
            return False
        return  self.attr==other.attr and self.value==other.value and self.op==other.op

    def __or__(self, other: Any):
        if not isinstance(other, AttributeFilter):
            raise Exception("Not an attribute filter. Can not run an OR operation.")
        return OrAttributeFilter(self, other)

    def __and__(self, other: Any):
        if not isinstance(other, AttributeFilter):
            raise Exception("Not an attribute filter. Can not run an AND operation.")
        return AndAttributeFilter(self, other)

    def match(self, layer_config: Dict[str, Any]):
        if self.attr in layer_config:
            return self.op(layer_config.get(self.attr), self.value)
        return False

    def op_as_str(self):
        raise Exception("Filter must implement op_as_str ")

    def __repr__(self):
        return f'{self.attr} {self.op_as_str()} {self.value}'


class Greater(AttributeFilter):
    def __init__(self, attr: str, value: Any):
        super().__init__(attr=attr, value=value, op=operator.gt)

    def op_as_str(self): return ">"


class GreaterEq(AttributeFilter):
    def __init__(self, attr: str, value: Any):
        super().__init__(attr=attr, value=value, op=operator.ge)

    def op_as_str(self): return ">="


class Smaller(AttributeFilter):
    def __init__(self, attr: str, value: Any):
        super().__init__(attr=attr, value=value, op=operator.lt)

    def op_as_str(self): return "<"


class SmallerEq(AttributeFilter):
    def __init__(self, attr: str, value: Any):
        super().__init__(attr=attr, value=value, op=operator.le)

    def op_as_str(self): return "<="


class NotEq(AttributeFilter):
    def __init__(self, attr: str, value: Any):
        super().__init__(attr=attr, value=value, op=operator.ne)

    def op_as_str(self): return "!="

class Eq(AttributeFilter):
    def __init__(self, attr: str, value: Any):
        super().__init__(attr=attr, value=value, op=operator.eq)

    def op_as_str(self): return "="