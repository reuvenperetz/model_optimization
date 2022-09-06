


class WrapperQuantizeConfig:

    def __init__(self,
                 is_weight_quantized: bool,
                 is_activation_quantized: bool,
                 requires_grad: bool = False,
                 store_float_weight: bool = False
                 ):

        self.is_weight_quantized = is_weight_quantized
        self.is_activation_quantized = is_activation_quantized
        self.requires_grad = requires_grad
        self.store_float_weight = store_float_weight

    def get_weight_quantizers(self):
        raise NotImplemented


    def get_activation_quantizers(self):
        raise NotImplemented



