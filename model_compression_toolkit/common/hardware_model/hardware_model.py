import pprint

from model_compression_toolkit.common.hardware_model.global_config import GlobalConfig
from model_compression_toolkit.common.hardware_model.operators import OperatorsSetBase
from model_compression_toolkit.common.hardware_model.quantization_config import OpQuantizationConfig, \
    QuantizationConfigOptions
from model_compression_toolkit.common.hardware_model.fusing import Fusing
from model_compression_toolkit.common.hardware_model.hardware_model_component import \
    HardwareModelComponent
from model_compression_toolkit.common.immutable import ImmutableClass
from model_compression_toolkit.common.hardware_model.current_hardware_model import _current_hardware_model, \
    get_current_model


def get_default_quantization_config_options():
    return get_current_model().default_qco


def get_default_quantization_config():
    assert len(get_current_model().default_qco.quantization_config_list) == 1
    return get_current_model().default_qco.quantization_config_list[0]


class HardwareModel(ImmutableClass):

    def __init__(self, default_qco, name="base"):
        super().__init__()
        self.name = name
        self.operator_set = []
        assert isinstance(default_qco, QuantizationConfigOptions)
        assert len(default_qco.quantization_config_list) == 1, f'Default QuantizationConfigOptions must contain only one option'
        self.default_qco = default_qco
        self.fusing_patterns = []
        self.global_config = GlobalConfig()

    def get_config_options_by_operators_set(self, operators_set_name: str):
        for op_set in self.operator_set:
            if operators_set_name == op_set.name:
                return op_set.qc_options
        return None

    def is_opset_in_model(self, opset_name: str):
        return opset_name in [x.name for x in self.operator_set]

    def append_component(self, hm_component: HardwareModelComponent):
        if isinstance(hm_component, Fusing):
            self.fusing_patterns.append(hm_component)
        elif isinstance(hm_component, OperatorsSetBase):
            self.operator_set.append(hm_component)
        else:
            raise Exception(f'Trying to append an unfamiliar HardwareModelComponent of type: {type(hm_component)}')

    def __enter__(self):
        _current_hardware_model.set(self)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_value is not None:
            print(exc_value, exc_value.args)
            raise exc_value
        _current_hardware_model.reset()
        self.initialized_done()
        return self

    def get_default_config(self) -> OpQuantizationConfig:
        return self.default_qco.quantization_config_list[0]

    def get_info(self):
        return {"Model name": self.name,
                "Default quantization config": self.get_default_config().get_info(),
                "Operators sets": [o.get_info() for o in self.operator_set],
                "Fusing patterns": [f.get_info() for f in self.fusing_patterns]
                }

    def show(self):
        pprint.pprint(self.get_info(), sort_dicts=False)
