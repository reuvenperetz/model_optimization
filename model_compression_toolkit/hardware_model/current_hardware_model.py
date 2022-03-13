
class CurrentHardwareModel:
    """A thread-local stack of objects for providing implicit defaults."""

    def __init__(self):
        super(CurrentHardwareModel, self).__init__()
        self.hwm = None

    def get(self):
        if self.hwm is None:
            raise Exception()
        return self.hwm

    def reset(self):
        self.hwm = None

    def set(self, hwm):
        self.hwm = hwm


_current_hardware_model = CurrentHardwareModel()

def get_current_model():
    return _current_hardware_model.get()
