# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from model_compression_toolkit import hardware_representation as hw_model


def get_qnnpack_model():
    # Create a quantization config.
    # A quantization configuration defines how an operation
    # should be quantized on the modeled hardware:
    eight_bits = hw_model.OpQuantizationConfig(
        activation_quantization_method=hw_model.QuantizationMethod.UNIFORM,
        weights_quantization_method=hw_model.QuantizationMethod.SYMMETRIC,
        activation_n_bits=8,
        weights_n_bits=8,
        weights_per_channel_threshold=False,
        enable_weights_quantization=True,
        enable_activation_quantization=True
    )

    # Create a QuantizationConfigOptions, which defines a set
    # of possible configurations to consider when quantizing a set of operations (in mixed-precision, for example).
    # If the QuantizationConfigOptions contains only one configuration,
    # this configuration will be used for the operation quantization:
    default_configuration_options = hw_model.QuantizationConfigOptions([eight_bits])

    # Create a HardwareModel and set its default quantization config.
    # This default configuration will be used for all operations
    # unless specified otherwise (see OperatorsSet, for example):
    qnnpack_model = hw_model.HardwareModel(default_configuration_options, name='qnnpack')

    # To start defining the model's components (such as operator sets, and fusing patterns),
    # use 'with' the hardware model instance, and create them as below:
    with qnnpack_model:
        # Fusing: [Conv, Relu], [Conv, BatchNorm], [Conv, BatchNorm, Relu], [Linear, Relu]
        conv = hw_model.OperatorsSet("Conv")
        batchnorm = hw_model.OperatorsSet("BatchNorm")
        relu = hw_model.OperatorsSet("Relu")
        linear = hw_model.OperatorsSet("Linear")

        hw_model.Fusing([conv, batchnorm, relu])
        hw_model.Fusing([conv, batchnorm])
        hw_model.Fusing([conv, relu])
        hw_model.Fusing([linear, relu])

    return qnnpack_model


if __name__ == '__main__':
    qnnpack = get_qnnpack_model()
    qnnpack.show()

#
# class HistogramObserver(_ObserverBase):
#     r"""
#     The module records the running histogram of tensor values along with
#     min/max values. ``calculate_qparams`` will calculate scale and zero_point.
#     Args:
#         bins: Number of bins to use for the histogram
#         upsample_rate: Factor by which the histograms are upsampled, this is
#                        used to interpolate histograms with varying ranges across observations
#         dtype: Quantized data type
#         qscheme: Quantization scheme to be used
#         reduce_range: Reduces the range of the quantized data type by 1 bit
#     The scale and zero point are computed as follows:
#     1. Create the histogram of the incoming inputs.
#         The histogram is computed continuously, and the ranges per bin change
#         with every new tensor observed.
#     2. Search the distribution in the histogram for optimal min/max values.
#         The search for the min/max values ensures the minimization of the
#         quantization error with respect to the floating point model.
#     3. Compute the scale and zero point the same way as in the
#         :class:`~torch.ao.quantization.MinMaxObserver`
#
# def get_default_qconfig(backend='fbgemm'):
#     """
#     Returns the default PTQ qconfig for the specified backend.
#     Args:
#       * `backend`: a string representing the target backend. Currently supports `fbgemm`
#         and `qnnpack`.
#     Return:
#         qconfig
#     """
#
#     if backend == 'fbgemm':
#         qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
#                           weight=default_per_channel_weight_observer)
#     elif backend == 'qnnpack':
#         qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
#                           weight=default_weight_observer)
#     else:
#         qconfig = default_qconfig
#     return qconfig
# default_weight_observer = MinMaxObserver.with_args(
#     dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
# )
#
# Fuse modules: combine operations/modules into a single module to obtain higher
# accuracy and performance. This is done using the torch.quantization.fuse_modules() API,
# which takes in lists of modules to be fused. We currently support the following fusions:
# [Conv, Relu], [Conv, BatchNorm], [Conv, BatchNorm, Relu], [Linear, Relu]
