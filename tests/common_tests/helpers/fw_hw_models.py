from model_compression_toolkit import FrameworkHardwareModel
from model_compression_toolkit.hardware_model import HardwareModel
from model_compression_toolkit.hardware_model.quantization_config import QuantizationMethod, OpQuantizationConfig, \
    QuantizationConfigOptions


def get_float_fw_hw_model():
    float_config = OpQuantizationConfig(
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=False,
        enable_activation_quantization=False
    )
    default_configuration_options = QuantizationConfigOptions([float_config])
    return FrameworkHardwareModel(HardwareModel(default_configuration_options, name='float_hw_model'))


def get_2bits_fw_hw_model():
    config = OpQuantizationConfig(
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=2,
        weights_n_bits=2,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        enable_activation_quantization=True
    )
    default_configuration_options = QuantizationConfigOptions([config])
    return FrameworkHardwareModel(HardwareModel(default_configuration_options, name='2bits_hw_model'))



def get_4bits_fw_hw_model():
    config = OpQuantizationConfig(
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=4,
        weights_n_bits=4,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        enable_activation_quantization=True
    )
    default_configuration_options = QuantizationConfigOptions([config])
    return FrameworkHardwareModel(HardwareModel(default_configuration_options, name='4bits_hw_model'))




def get_8bits_fw_hw_model():
    config = OpQuantizationConfig(
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        enable_activation_quantization=True
    )
    default_configuration_options = QuantizationConfigOptions([config])
    return FrameworkHardwareModel(HardwareModel(default_configuration_options, name='8bits_hw_model'))


def get_16bits_fw_hw_model():
    config = OpQuantizationConfig(
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=16,
        weights_n_bits=16,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        enable_activation_quantization=True
    )
    default_configuration_options = QuantizationConfigOptions([config])
    return FrameworkHardwareModel(HardwareModel(default_configuration_options, name='16bits_hw_model'))


def get_32bits_fw_hw_model():
    config = OpQuantizationConfig(
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=32,
        weights_n_bits=32,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        enable_activation_quantization=True
    )
    default_configuration_options = QuantizationConfigOptions([config])
    return FrameworkHardwareModel(HardwareModel(default_configuration_options, name='32bits_hw_model'))


