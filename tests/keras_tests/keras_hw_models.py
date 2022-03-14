# from model_compression_toolkit.common.hardware_model.framework_hardware_model import \
#     FrameworkHardwareModel
# from tests.common_tests.helpers.hw_models import get_float_hw_model, get_16bits_hw_model
#
#
# def get_keras_float_model():
#     float_model = get_float_hw_model()
#     float_fw = FrameworkHardwareModel(float_model,
#                                          name='float_fw')
#     return float_fw
#
#
#
# def get_keras_16bits_model():
#     sixteen_bits_model = get_16bits_hw_model()
#     sixteen_bits_fw = FrameworkHardwareModel(sixteen_bits_model,
#                                                 name='16bits_fw')
#     return sixteen_bits_fw