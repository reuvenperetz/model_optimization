
class ImmutableClass(object):

    _initialized = False

    def __init__(self):
        self._initialized = False

    def __setattr__(self, *args, **kwargs):
        if self._initialized:
            raise Exception('Immutable Class')
        else:
            object.__setattr__(self, *args, **kwargs)

    def initialized_done(self):
        if self._initialized:
            raise Exception('reinitialized')
        self._initialized = True