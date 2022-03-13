import unittest

from keras.layers import Conv2D, Conv2DTranspose, ReLU

# from hardware_modeling.model2framework.framework_hardware_model.attribute_filter import Greater, Smaller, GreaterEq, Eq
# from keras_modeling.keras_layer_filter_params import KerasLayerFilterParams

from model_compression_toolkit.common.hardware_model.attribute_filter import GreaterEq, Smaller, Greater, Eq
from model_compression_toolkit.keras.hardware_model.keras_layer_filter_params import KerasLayerFilterParams


class LayerWithParamsTest(unittest.TestCase):

    def test_keras_layers_with_params(self):
        conv_with_params = KerasLayerFilterParams(Conv2D,
                                                  Greater("filters", 2),
                                                  Smaller("filters", 4),
                                                  activation='softmax',
                                                  kernel_size=(3, 4),
                                                  filters=3)

        conv = Conv2D(filters=3, kernel_size=(3, 4), activation='softmax')
        self.assertTrue(conv_with_params.match(conv))
        conv = Conv2D(filters=2, kernel_size=(3, 4), activation='softmax')
        self.assertFalse(conv_with_params.match(conv))
        conv = Conv2DTranspose(filters=3, kernel_size=(3, 4), activation='softmax')
        self.assertFalse(conv_with_params.match(conv))

        relu_with_params = KerasLayerFilterParams(ReLU, GreaterEq("max_value", 0.5) | Smaller("max_value", 0.2))
        self.assertTrue(relu_with_params.match(ReLU(max_value=0.1)))
        self.assertTrue(relu_with_params.match(ReLU(max_value=0.5)))
        self.assertFalse(relu_with_params.match(ReLU(max_value=0.3)))

        relu_with_params = KerasLayerFilterParams(ReLU, Eq("max_value", None) | Eq("max_value", 6))
        self.assertTrue(relu_with_params.match(ReLU()))
        self.assertTrue(relu_with_params.match(ReLU(max_value=6)))
        self.assertFalse(relu_with_params.match(ReLU(max_value=8)))