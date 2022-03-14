from typing import Any

from model_compression_toolkit.hardware_model.layer_filter_params import LayerFilterParams

LayerFilterParams


class KerasLayerFilterParams(LayerFilterParams):
    def __init__(self, layer: Any, *conditions, **kwargs):
        super().__init__(layer,
                         conditions,
                         **kwargs)

    @staticmethod
    def get_layer_attributes(layer):
        return layer.get_config()


