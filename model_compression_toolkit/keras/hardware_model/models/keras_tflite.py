import tensorflow as tf
from keras.layers import Conv2D, Dense, Reshape, ZeroPadding2D, \
    MaxPooling2D, ReLU, AveragePooling2D, Activation, DepthwiseConv2D

from keras import layers
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from model_compression_toolkit.common.hardware_model.attribute_filter import Eq

from model_compression_toolkit.common.hardware_model.framework_hardware_model import \
    FrameworkHardwareModel
from model_compression_toolkit.common.hardware_model.operations_to_layers import OperationsSetToLayers

from model_compression_toolkit.common.hardware_model.models_dictionary import hw_models_dict, HWModelCfg
from model_compression_toolkit.keras.keras_layer_filter_params import KerasLayerFilterParams


def get_keras_hardware_model_tflite():
    tflite_hm = hw_models_dict.get(HWModelCfg.TFLITE)
    tflite_keras = FrameworkHardwareModel(tflite_hm, name='tflite_keras')

    with tflite_keras:
        OperationsSetToLayers("PreserveQuantizationParams", [AveragePooling2D,
                                                             tf.nn.avg_pool2d,
                                                             layers.Concatenate,
                                                             tf.concat,
                                                             MaxPooling2D,
                                                             layers.Multiply,
                                                             tf.multiply,
                                                             Reshape,
                                                             tf.reshape,
                                                             KerasLayerFilterParams(tf.image.resize, method=ResizeMethod.BILINEAR),
                                                             tf.nn.space_to_depth,
                                                             ZeroPadding2D,
                                                             tf.gather,
                                                             tf.compat.v1.batch_to_space_nd,
                                                             tf.space_to_batch_nd,
                                                             tf.transpose,
                                                             tf.maximum,
                                                             layers.Maximum,
                                                             tf.minimum,
                                                             layers.Minimum,
                                                             tf.pad,
                                                             tf.slice,
                                                             layers.SlicingOpLambda])

        OperationsSetToLayers("FullyConnected", [Dense])
        OperationsSetToLayers("L2Normalization", [tf.math.l2_normalize])
        OperationsSetToLayers("LogSoftmax", [tf.nn.log_softmax])
        OperationsSetToLayers("Tanh", [tf.nn.tanh,
                                       KerasLayerFilterParams(Activation, activation="tanh")])

        OperationsSetToLayers("Softmax", [tf.nn.softmax,
                                          layers.Softmax,
                                          KerasLayerFilterParams(Activation, activation="softmax")])

        OperationsSetToLayers("Logistic", [tf.sigmoid,
                                           KerasLayerFilterParams(Activation, activation="sigmoid")])

        OperationsSetToLayers("Conv2d", [Conv2D])
        OperationsSetToLayers("DepthwiseConv2D", [DepthwiseConv2D])

        OperationsSetToLayers("Relu", [tf.nn.relu,
                                       tf.nn.relu6,
                                       KerasLayerFilterParams(ReLU, Eq("max_value", None) | Eq("max_value", 6)),
                                       KerasLayerFilterParams(Activation, activation="relu")])



        OperationsSetToLayers("Elu", [tf.nn.elu,
                                      KerasLayerFilterParams(Activation, activation="relu")])

        OperationsSetToLayers("BatchNorm", [layers.BatchNormalization,
                                            tf.nn.batch_normalization])

        OperationsSetToLayers("Squeeze", [tf.squeeze])
        OperationsSetToLayers("BiasAdd", [tf.nn.bias_add])
        OperationsSetToLayers("Add", [tf.add,
                                      layers.Add])

    return tflite_keras


if __name__ == "__main__":
    model = get_keras_hardware_model_tflite()
    model.show()
