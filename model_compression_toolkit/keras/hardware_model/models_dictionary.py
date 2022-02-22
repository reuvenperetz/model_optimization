from enum import Enum

from model_compression_toolkit.keras.hardware_model.models.keras_imx500 import get_imx500_keras
from model_compression_toolkit.keras.hardware_model.models.keras_tflite import get_keras_hardware_model_tflite


class KerasHWModelCfg(Enum):
    IMX500 = 0
    TFLITE = 2


hw_models_dict = {KerasHWModelCfg.IMX500: get_imx500_keras(),
                  KerasHWModelCfg.TFLITE: get_keras_hardware_model_tflite()}
