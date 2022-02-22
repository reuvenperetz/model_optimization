
class _CurrentFrameworkHardwareModel(object):
    """A thread-local stack of objects for providing implicit defaults."""

    def __init__(self):
        super(_CurrentFrameworkHardwareModel, self).__init__()
        self.fwhw_model = None

    def get(self):
        if self.fwhw_model is None:
            raise Exception()
        return self.fwhw_model

    def reset(self):
        self.fwhw_model = None

    def set(self, fwhw_model):
        self.fwhw_model = fwhw_model


_current_framework_hardware_model = _CurrentFrameworkHardwareModel()
