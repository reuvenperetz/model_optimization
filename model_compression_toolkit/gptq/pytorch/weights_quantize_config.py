from model_compression_toolkit.core.pytorch.back2framework.quantization_wrapper.wrapper_quantize_config import \
    WrapperQuantizeConfig




class WeightsQuantizeConfig(WrapperQuantizeConfig):

    def __init__(self,
                 weight_quantizers:list):
        super().__init__(is_weight_quantized=True,
                         is_activation_quantized=False,
                         requires_grad=True,
                         store_float_weight=True)

        self._weight_quantizers = weight_quantizers

    def get_weight_quantizers(self):
        return self._weight_quantizers

    def get_activation_quantizer(self):
        return []



