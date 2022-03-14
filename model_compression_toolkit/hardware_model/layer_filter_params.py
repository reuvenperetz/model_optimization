from typing import Any, Dict


class LayerFilterParams:
    def __init__(self, layer: Any, *conditions, **kwargs):
        self.layer = layer
        self.conditions = conditions[0]
        self.kwargs = kwargs
        self.__name__ = self.create_name()
        # self.__name__ = layer.__name__+str(kwargs)

    def create_name(self):
        params = [f'{k}={v}' for k,v in self.kwargs.items()]
        params.extend([str(c) for c in self.conditions])
        params_str = ', '.join(params)
        return f'{self.layer.__name__}({params_str})'


    def get_layer_attributes(self, layer):
        raise Exception('Framework layer should implement a method to get a layer attributes')

    def match(self, layer: Any):
        if self.layer != type(layer):
            return False
        layer_config = self.get_layer_attributes(layer)
        for attr, value in self.kwargs.items():
            if layer_config.get(attr) is None:
                return False
            if layer_config.get(attr) != value:
                return False
        for c in self.conditions:
            if not c.match(layer_config):
                return False
        return True
