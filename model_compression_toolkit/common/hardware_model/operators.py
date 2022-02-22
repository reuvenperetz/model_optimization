# from hardware_modeling import QuantizationConfigOptions
from model_compression_toolkit.common.hardware_model.current_hardware_model import _current_hardware_model
# from hardware_modeling.hardware_definition.hardware_model.hardware_model_component import HardwareModelComponent
from model_compression_toolkit.common.hardware_model.hardware_model_component import \
    HardwareModelComponent
from model_compression_toolkit.common.hardware_model.quantization_config \
    import \
    QuantizationConfigOptions


class OperatorsSetBase(HardwareModelComponent):
    """
    Base class to represent a set of operators.
    """
    def __init__(self, name):
        super().__init__(name=name)


class OperatorsSet(OperatorsSetBase):
    def __init__(self, name: str, qc_options: QuantizationConfigOptions=None):
        """
        Set of operators that are represented by a unique label.
        Args:
            name: Name of the set.
            qc_options: Configuration options to use for this set of operations.
        """

        super().__init__(name)
        self.qc_options = qc_options
        is_fusing_set = qc_options is None
        self.is_default = _current_hardware_model.get().default_qco == self.qc_options or is_fusing_set


    def get_info(self):
        """

        Returns: Info about the set as a dictionary.

        """
        return {"name": self.name,
                "is_default_qc": self.is_default}


class OperatorSetConcat(OperatorsSetBase):
    """
    Concatenate a list of operator sets to treat them similarly in different places (like fusing).
    """
    def __init__(self, *args):
        """
        Group a list of operation sets and create a new one.
        Args:
            *args: List of operator sets to group.
        """
        name = "_".join([a.name for a in args])
        super().__init__(name=name)
        self.op_set_list = args

    def get_info(self):
        """

        Returns: Info about the sets group as a dictionary.

        """
        return {"name": self.name,
                "ops_set_list": [s.name for s in self.op_set_list]}
