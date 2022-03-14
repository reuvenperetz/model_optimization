from model_compression_toolkit.hardware_model.current_hardware_model import _current_hardware_model


class HardwareModelComponent:
    def __init__(self, name):
        self.name = name
        _current_hardware_model.get().append_component(self)

    def get_info(self):
        return {}
