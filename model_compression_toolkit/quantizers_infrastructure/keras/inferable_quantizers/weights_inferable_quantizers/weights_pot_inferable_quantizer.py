# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

import numpy as np

from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import QuantizationTarget

if FOUND_TF:
    import tensorflow as tf
    from tensorflow_model_optimization.python.core.quantization.keras import quant_ops
    from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers.base_pot_inferable_quantizer import BasePOTInferableQuantizer

    class WeightsPOTInferableQuantizer(BasePOTInferableQuantizer):
        """
        Class for quantizing weights using power-of-two quantizer
        """

        def __init__(self,
                     num_bits: int,
                     threshold: np.ndarray,
                     signed: bool,
                     per_channel: bool,
                     channel_axis: int,
                     input_num_dims: int):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing activations
                signed: whether or not to use signed quantization
                per_channel: whether to use per-channel quantization
                channel_axis: axis along which to apply per-channel quantization
            """
            # Call the superclass constructor with the given parameters, along with the target of Weights quantization
            super(WeightsPOTInferableQuantizer, self).__init__(num_bits=num_bits,
                                                               threshold=threshold,
                                                               signed=signed,
                                                               quantization_target=QuantizationTarget.Weights)

            self.per_channel = per_channel
            self.channel_axis = channel_axis

            # Get the shape of the threshold array
            self.threshold_shape = np.asarray(threshold).shape

            # Tensorflow's fake_quant_with_min_max_vars_per_channel only works on last axis, so
            # need to move the quantization axis to the last axis
            if per_channel and channel_axis not in [-1, input_num_dims - 1]:
                # If per-channel quantization is being used and the channel axis is not the last axis,
                # create a permutation vector to move the channel axis to the last position
                self.perm_vec = list(np.arange(input_num_dims))
                self.perm_vec[channel_axis] = input_num_dims - 1
                self.perm_vec[input_num_dims - 1] = channel_axis
            else:
                # If per-channel quantization is not being used or the channel axis is already the last axis,
                # set the permutation vector to None
                self.perm_vec = None

            # self.min_range = tf.Variable(self.min_range.flatten(), dtype=tf.float32)
            # self.max_range = tf.Variable(self.max_range.flatten(), dtype=tf.float32)
            # thresholds = np.arange(3)
            #
            # print(thresholds)
            # self.min_range = -thresholds.flatten()
            # self.max_range = (thresholds - thresholds / 128).flatten()

        def __call__(self, inputs: tf.Tensor):
            """
            Quantize the given inputs using the quantizer parameters.

            Args:
                inputs: input tensor to quantize

            Returns:
                quantized tensor.
            """

            # If per-channel quantization is being used
            if self.per_channel:
                # If a permutation vector has been created to move the channel axis to the last position
                if self.perm_vec:
                    # Transpose the input tensor to move the channel axis to the last position
                    inputs = tf.transpose(inputs, perm=self.perm_vec)

                #
                # q_tensor = quant_ops.LastValueQuantize(
                #     inputs,
                #     # tf.Tensor(self.min_range.flatten()),
                #     # tf.Tensor(self.max_range.flatten()),
                #     self.min_range,
                #     self.max_range,
                #     is_training=False,
                #     num_bits=self.num_bits,
                #     per_channel=self.per_channel,
                #     symmetric=True,
                #     narrow_range=False
                # )

                # Quantize the input tensor using per-channel quantization
                q_tensor = tf.quantization.fake_quant_with_min_max_vars_per_channel(inputs,
                                                                                    min=self.min_range.flatten(),
                                                                                    max=self.max_range.flatten(),
                                                                                    num_bits=self.num_bits)
                if self.perm_vec:
                    # Transpose the quantized tensor back to its original shape
                    q_tensor = tf.transpose(q_tensor, perm=self.perm_vec)

                # Return the quantized tensor
                return q_tensor
            else:
                # If per-channel quantization is not being used, quantize the input tensor using regular quantization
                return tf.quantization.fake_quant_with_min_max_vars(inputs,
                                                                    min=self.min_range,
                                                                    max=self.max_range,
                                                                    num_bits=self.num_bits)

        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.

            Returns:
                Dictionary with the following keys: 'num_bits', 'signed', 'threshold', 'per_channel', 'channel_axis'
            """
            return {'num_bits': self.num_bits,
                    'signed': self.signed,
                    'threshold': self.threshold,
                    'per_channel': self.per_channel,
                    'channel_axis': self.channel_axis}

else:
    class WeightsPOTInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using WeightsPOTInferableQuantizer. '
                            'Could not find Tensorflow package.')
