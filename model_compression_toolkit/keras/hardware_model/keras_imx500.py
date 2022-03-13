
import tensorflow as tf
from keras.layers import Conv2D, Dense, Reshape, Flatten, Cropping2D, ZeroPadding2D, \
    Dropout, MaxPooling2D, ReLU, Activation, Add, PReLU, BatchNormalization, Conv2DTranspose, DepthwiseConv2D

# from examples.imx500 import get_imx500_model
# from hardware_modeling.model2framework.framework_hardware_model.framework_hardware_model import FrameworkHardwareModel
# from hardware_modeling.model2framework.operations_to_layers import OperationsSetToLayers
# from keras_modeling.keras_layer_filter_params import KerasLayerFilterParams
# from tensorflow.python.keras.engine.base_layer import TensorFlowOpLayer

from model_compression_toolkit.hardware_model.models import get_imx500_model
from model_compression_toolkit.common.hardware_model.framework_hardware_model import \
    FrameworkHardwareModel
from model_compression_toolkit.common.hardware_model.operations_to_layers import \
    OperationsSetToLayers
from model_compression_toolkit.keras.hardware_model.keras_layer_filter_params import KerasLayerFilterParams


def get_imx500_keras():
    imx500hm = get_imx500_model()
    imx500_keras = FrameworkHardwareModel(imx500hm,
                                          name='imx500_keras')

    with imx500_keras:
        OperationsSetToLayers("NoQuantization", [Reshape,
                                                 tf.reshape,
                                                 Flatten,
                                                 Cropping2D,
                                                 ZeroPadding2D,
                                                 Dropout,
                                                 MaxPooling2D,
                                                 tf.split,
                                                 tf.quantization.fake_quant_with_min_max_vars,
                                                 BatchNormalization])

        OperationsSetToLayers("Conv", [Conv2D,
                                       DepthwiseConv2D])

        OperationsSetToLayers("FullyConnected", [Dense])

        OperationsSetToLayers("ConvTranspose", [Conv2DTranspose])

        OperationsSetToLayers("AnyReLU", [tf.nn.relu,
                                          tf.nn.relu6,
                                          KerasLayerFilterParams(ReLU, negative_slope=0.0),
                                          KerasLayerFilterParams(Activation, activation="relu")])

        OperationsSetToLayers("Add", [tf.add,
                                      Add])

        OperationsSetToLayers("PReLU", [PReLU])

        OperationsSetToLayers("Swish", [tf.nn.swish,
                                        KerasLayerFilterParams(Activation, activation="swish")])

        OperationsSetToLayers("Sigmoid", [tf.nn.sigmoid,
                                          KerasLayerFilterParams(Activation, activation="sigmoid")])

        OperationsSetToLayers("Tanh", [tf.nn.tanh,
                                       KerasLayerFilterParams(Activation, activation="tanh")])

        # OperationsSetToLayers("NoWeightsQuantization", [InputLayer,
        #                                                 Activation,
        #                                                 Concatenate,
        #                                                 GlobalAveragePooling2D])
        # TODO: framework and hardware versions

    return imx500_keras


DEFAULT_KERAS_IMX500_MODEL = get_imx500_keras()
