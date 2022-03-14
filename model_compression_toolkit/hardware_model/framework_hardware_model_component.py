
from model_compression_toolkit.hardware_model.current_framework_hardware_model import  _current_framework_hardware_model


class FrameworkHardwareModelComponent:
    def __init__(self, name: str):
        self.name = name
        _current_framework_hardware_model.get().append_component(self)

    def get_info(self):
        return {}
