import keras
from keras.applications import MobileNetV2

from tests.keras_tests.exporter_tests.tflite_int8_exporter_base_test import TFLiteINT8ExporterBaseTest

layers = keras.layers





class TestConv2DExporter(TFLiteINT8ExporterBaseTest):
    def get_model(self):
        return self.get_one_layer_model(layers.Conv2D(6,5))

    # def run_checks(self):

class TestDenseExporter(TFLiteINT8ExporterBaseTest):
    def get_model(self):
        return self.get_one_layer_model(layers.Dense(20))

    def get_input_shape(self):
        return (3,4,5,6,7)


class TestReluExporter(TFLiteINT8ExporterBaseTest):
    def get_model(self):
        return self.get_one_layer_model(layers.ReLU())

class TestMBV2Exporter(TFLiteINT8ExporterBaseTest):
    def get_model(self):
        return MobileNetV2()
