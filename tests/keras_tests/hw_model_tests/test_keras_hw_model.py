import unittest
from keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np

from model_compression_toolkit import keras_post_training_quantization
from model_compression_toolkit.keras.hardware_model.models.keras_imx500 import get_imx500_keras


class KerasHWModelTest(unittest.TestCase):

    def test_keras_imx500_model(self):
        fw_hw_model = get_imx500_keras()
        model = MobileNetV2()

        def rep_data():
            return [np.random.randn(1, 224, 224, 3)]

        quantized_model, _ = keras_post_training_quantization(model, rep_data, n_iter=1, fw_hw_model=fw_hw_model)
