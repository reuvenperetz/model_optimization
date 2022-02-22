from enum import Enum

from model_compression_toolkit.models.imx500 import get_imx500_model
from model_compression_toolkit.models import get_qnnpack_model
from model_compression_toolkit.models.tflite import get_tflite_hw_model


class HWModelCfg(Enum):
    IMX500 = 0
    QNNPACK = 1
    TFLITE = 2


hw_models_dict = {HWModelCfg.IMX500: get_imx500_model(),
                  HWModelCfg.QNNPACK: get_qnnpack_model(),
                  HWModelCfg.TFLITE: get_tflite_hw_model()}
