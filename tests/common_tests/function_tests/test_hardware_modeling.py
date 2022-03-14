import unittest

from model_compression_toolkit import QuantizationMethod
from model_compression_toolkit.hardware_model.hardware_model import \
    HardwareModel, get_default_quantization_config_options
from model_compression_toolkit.hardware_model.models.imx500 import get_imx500_model
from model_compression_toolkit.hardware_model.quantization_config \
    import \
    QuantizationConfigOptions, OpQuantizationConfig



default_qc = OpQuantizationConfig(enable_activation_quantization=True,
                                  enable_weights_quantization=True,
                                  activation_n_bits=8,
                                  weights_n_bits=8,
                                  weights_per_channel_threshold=True,
                                  activation_quantization_method=QuantizationMethod.LUT_QUANTIZER,
                                  weights_quantization_method=QuantizationMethod.LUT_QUANTIZER)
default_qc_options = QuantizationConfigOptions(default_qc)

class HardwareModelingTest(unittest.TestCase):

    def test_immutable_hwm(self):
        model = get_imx500_model()
        with self.assertRaises(Exception) as e:
            model.fusing_patterns = 0
        self.assertEqual('Immutable Class', str(e.exception))

    def test_get_default_options(self):
        default = default_qc_options
        with HardwareModel(default):
            self.assertEqual(get_default_quantization_config_options(), default)
