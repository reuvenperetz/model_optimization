import itertools
import pprint

# from hardware_modeling import HardwareModel
# from hardware_modeling.immutable import ImmutableClass
from model_compression_toolkit.common.hardware_model.operators import OperatorsSet


from model_compression_toolkit.common.hardware_model.hardware_model import \
    HardwareModel
from model_compression_toolkit.common.hardware_model.current_framework_hardware_model import _current_framework_hardware_model

# from hardware_modeling.model2framework.framework_hardware_model.framework_hardware_model_component import \
#     FrameworkHardwareModelComponent
from model_compression_toolkit.common.hardware_model.framework_hardware_model_component import \
    FrameworkHardwareModelComponent
from model_compression_toolkit.common.hardware_model.layer_filter_params import \
    LayerFilterParams
from model_compression_toolkit.common.hardware_model.operations_to_layers import OperationsToLayers, OperationsSetToLayers

from model_compression_toolkit.common.immutable import ImmutableClass


class FrameworkHardwareModel(ImmutableClass):
    def __init__(self,
                 hw_model: HardwareModel,
                 # activation_quantizer_mapping: Dict[QuantizationMethod, Callable],
                 # weights_quantizer_mapping: Dict[QuantizationMethod, Callable],
                 name: str="base"):

        super().__init__()
        self.name = name
        assert isinstance(hw_model, HardwareModel)
        self.hw_model = hw_model
        self.op_sets_to_layers = OperationsToLayers()
        self.layer2qco, self.filterlayer2qco = {}, {}
        # self.activation_quantizer_mapping = activation_quantizer_mapping
        # self.weights_quantizer_mapping = weights_quantizer_mapping
        self.hwm_opsets_not_used = [s.name for s in hw_model.operator_set]
        self.remove_fusing_names_from_not_used_list()

    def get_op_layers(self, op: OperatorsSet):
        return self.op_sets_to_layers.get_layers_by_op(op)

    def get_fusing_patterns(self):
        res = []
        for p in self.hw_model.fusing_patterns:
            ops = [self.get_op_layers(x) for x in p.operator_groups_list]
            # for op in p.operator_groups_list:
            res.extend(itertools.product(*ops))
        return [list(x) for x in res]




    def get_info(self):
        return {"Framework Hardware Model": self.name,
                "Hardware model": self.hw_model.get_info(),
                "Operations to layers": {op2layer.name:[l.__name__ for l in op2layer.layers] for op2layer in self.op_sets_to_layers.op_sets_to_layers}}


    def show(self):
        pprint.pprint(self.get_info(), sort_dicts=False, width=110)




    def remove_fusing_names_from_not_used_list(self):
        for f in self.hw_model.fusing_patterns:
            for s in f.operator_groups_list:
                self.remove_opset_from_not_used_list(s.name)

    def remove_opset_from_not_used_list(self, opset_to_remove: str):
        if opset_to_remove in self.hwm_opsets_not_used:
            self.hwm_opsets_not_used.remove(opset_to_remove)


    def append_component(self, hm_component: FrameworkHardwareModelComponent):
        if isinstance(hm_component, OperationsSetToLayers):
            self.op_sets_to_layers += hm_component
        else:
            raise Exception(f'Trying to append an unfamiliar HardwareModelComponent of type: {type(hm_component)}')

    def raise_warnings(self):
        for op in self.hwm_opsets_not_used:
            print(f'Warning: {op} is defined in hardware model, but is not used in framework hardware model.')


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

    def is_layer_in_model(self, layer_type, layer):
        if layer_type in self.layer2qco:
            return True
        for fl, qco in self.filterlayer2qco.items():
            if fl.match(layer):
                return True
        return False

    def get_qco_by_layer(self, layer_type, layer):
        if layer_type in self.layer2qco:
            return self.layer2qco.get(layer_type)
        for fl, qco in self.filterlayer2qco.items():
            if fl.match(layer):
                return qco
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




