from model_compression_toolkit.common.hardware_model.quantization_config import QuantizationMethod
from model_compression_toolkit.common.hardware_model import OpQuantizationConfig, QuantizationConfigOptions
from model_compression_toolkit.common.hardware_model.fusing import Fusing
from model_compression_toolkit.hardware_model.hardware_model import \
    HardwareModel, get_default_quantization_config_options
from model_compression_toolkit.hardware_model.operators import OperatorsSet, OperatorSetConcat


def get_imx500_model():
    # Create a quantization config.
    # A quantization configuration defines how an operation
    # should be quantized on the modeled hardware:

    eight_bits = OpQuantizationConfig(
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        enable_activation_quantization=True
    )

    # Create a QuantizationConfigOptions, which defines a set
    # of possible configurations to consider when quantizing a set of operations (in mixed-precision, for example).
    # If the QuantizationConfigOptions contains only one configuration,
    # this configuration will be used for the operation quantization:

    default_configuration_options = QuantizationConfigOptions([eight_bits])

    # Create a HardwareModel and set its default quantization config.
    # This default configuration will be used for all operations
    # unless specified otherwise (see OperatorsSet, for example):

    imx500 = HardwareModel(default_configuration_options, name='imx500')

    # To start defining the model's components (such as operator sets, and fusing patterns),
    # use 'with' the hardware model instance, and create them as below:

    with imx500:
        # Create an OperatorsSet to represent a set of operations.
        # Each OperatorsSet has a unique label.
        # If a quantization configuration options is passed, these options will
        # be used for operations that will be attached to this set's label.
        # Otherwise, it will be a configure-less set (used in fusing):

        # May suit for operations like: Dropout, Reshape, etc.
        OperatorsSet("NoQuantization",
                              get_default_quantization_config_options().clone_and_edit(
                                  enable_weights_quantization=False,
                                  enable_activation_quantization=False))

        # May suit for operations like: BatchNormalization, PReLU, etc.
        # OperatorsSet("NoWeightsQuantization",
        #                       get_default_quantization_config_options().clone_and_edit(
        #                           enable_weights_quantization=False))

        # To quantize a model using mixed-precision, create
        # a QuantizationConfigOptions with more than one
        # QuantizationConfig.
        # In this example, we quantize some operations' weights
        # using 2, 4 or 8 bits, and when using 2 or 4 bits, it's possible
        # to quantize the operations' activations using LUT.
        four_bits = eight_bits.clone_and_edit(weights_n_bits=4)
        two_bits = eight_bits.clone_and_edit(weights_n_bits=2)
        four_bits_lut = four_bits.clone_and_edit(
            weights_quantization_method=QuantizationMethod.LUT_QUANTIZER)
        two_bits_lut = two_bits.clone_and_edit(
            weights_quantization_method=QuantizationMethod.LUT_QUANTIZER)

        mixed_precision_configuration_options = QuantizationConfigOptions([eight_bits,
                                                                           four_bits,
                                                                           two_bits,
                                                                           # four_bits_lut,
                                                                           # two_bits_lut
                                                                           ],
                                                                          base_config=eight_bits)


        # Define operator sets that use mixed_precision_configuration_options:
        OperatorsSet("ConvTranspose", mixed_precision_configuration_options)
        conv = OperatorsSet("Conv", mixed_precision_configuration_options)
        fc_fuse = OperatorsSet("FullyConnected", mixed_precision_configuration_options)

        # Define operations sets for without quantization configuration
        # options (useful for creating fusing patterns, for example):
        disable_weights_quantization_options = get_default_quantization_config_options().clone_and_edit(enable_weights_quantization=False)
        any_relu = OperatorsSet("AnyReLU", disable_weights_quantization_options)
        add = OperatorsSet("Add", disable_weights_quantization_options)
        prelu = OperatorsSet("PReLU", disable_weights_quantization_options)
        swish = OperatorsSet("Swish", disable_weights_quantization_options)
        sigmoid = OperatorsSet("Sigmoid", disable_weights_quantization_options)
        tanh = OperatorsSet("Tanh", disable_weights_quantization_options)

        # Define fusing patterns using the sets that were defined.
        # To group multiple sets with regard to fusing, an OperatorSetConcat can be created
        activations_after_conv_to_fuse = OperatorSetConcat(any_relu, swish, prelu, sigmoid, tanh)
        activations_after_fc_to_fuse = OperatorSetConcat(any_relu, swish, sigmoid)

        Fusing([conv, activations_after_conv_to_fuse])
        Fusing([fc_fuse, activations_after_fc_to_fuse])
        Fusing([conv, add, any_relu])
        Fusing([conv, any_relu, add])

    return imx500


