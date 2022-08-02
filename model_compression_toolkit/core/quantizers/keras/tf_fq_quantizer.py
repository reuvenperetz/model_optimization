import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer


class TFFakeQuantQuantizer(Quantizer):

    def __init__(self,
                 nbits,
                 min_range,
                 max_range,
                 channel_axis: int = -1,
                 per_channel: bool = True):

        self.nbits = nbits
        self.min_range = min_range
        self.max_range = max_range
        # self.min_range = tf.Variable(min_range,
        #                              trainable=False,
        #                              dtype=tf.float32)
        #
        # self.max_range = tf.Variable(max_range,
        #                              trainable=False,
        #                              dtype=tf.float32)

        self.channel_axis = channel_axis
        self.per_channel = per_channel
        self.delta = (self.max_range - self.min_range) / (2 ** self.nbits - 1)

    def get_config(self):
        return {"nbits": self.nbits,
                "min_range": self.min_range,
                "max_range": self.max_range,
                "channel_axis": self.channel_axis,
                "per_channel": self.per_channel}

    def build(self, tensor_shape, name, layer):
        return {}

    def __call__(self, inputs, training, weights, **kwargs):
        with tf.name_scope('TFFakeQuant'):
            return tf.quantization.fake_quant_with_min_max_args(inputs,
                                                         min=self.min_range,
                                                         max=self.max_range,
                                                         num_bits=self.nbits)


