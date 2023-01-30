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
import os
import tempfile
import unittest

import keras.models
import numpy as np
import tensorflow as tf
from keras import Input
from keras.applications import MobileNetV2
from keras.engine.base_layer import Layer
from keras.layers import Conv2D, BatchNormalization, ReLU, Dropout, Dense, Activation, DepthwiseConv2D

import model_compression_toolkit as mct
from model_compression_toolkit import keras_load_quantized_model
from model_compression_toolkit.exporter.model_exporter import keras_export_model, KerasExportMode, \
    tflite_export_model, \
    TFLiteExportMode
from model_compression_toolkit.exporter.model_wrapper import is_keras_layer_exportable
from tests.keras_tests.tpc_keras import get_activation_quantization_disabled_keras_tpc

_, SAVED_EXPORTABLE_MODEL_PATH_TF = tempfile.mkstemp('.h5')
_, SAVED_MODEL_PATH_INT8_TFLITE = tempfile.mkstemp('.tflite')
_, SAVED_MODEL_PATH_FQ_TFLITE = tempfile.mkstemp('.tflite')
_, SAVED_FLOAT_MODEL_PATH = tempfile.mkstemp('.h5')



def _get_model(input_shape):
    return MobileNetV2()
    inputs = Input(shape=input_shape)
    # x = Conv2D(7,7)(inputs)
    # x = BatchNormalization()(x)
    # x = ReLU(max_value=6)(x)
    # x = Conv2D(3,4)(x)
    # x = BatchNormalization()(x)
    # x = ReLU(max_value=6)(x)
    # # x = Conv2D(3, 5, padding="same")(inputs)
    # x = BatchNormalization()(x)
    # x = ReLU(max_value=6)(x)
    # x = Conv2D(5, 4, strides=2)(x)
    # x = BatchNormalization()(x)
    # x = ReLU(max_value=6)(x)
    # x = DepthwiseConv2D(8)(x)
    # x = BatchNormalization()(x)
    # x = ReLU(max_value=6)(x)
    # x = DepthwiseConv2D(9)(inputs)
    # x = Layer()(inputs)
    # x = BatchNormalization()(x)
    # x = ReLU(max_value=6)(x)
    # x = DepthwiseConv2D(10)(x)
    # x = BatchNormalization()(x)
    # x = ReLU(max_value=6)(x)
    # x = BatchNormalization()(x)
    # x = ReLU()(x)
    # x = Conv2D(3, 3)(x)
    # x = Activation('swish')(x)
    # x = Conv2D(3, 3, padding='same')(x)
    # x = ReLU(max_value=6)(x)
    # x = Dropout(rate=0.5)(x)
    x = Dense(10)(inputs)
    return keras.Model(inputs=inputs, outputs=x)


class TestTFLiteINT8Exporter(unittest.TestCase):

    # def tearDown(self):
    #     os.remove(SAVED_MODEL_PATH_TF)
    #     os.remove(SAVED_MODEL_PATH_TFLITE)
    #     os.remove(SAVED_EXPORTABLE_MODEL_PATH_TF)

    def setUp(self) -> None:
        input_shape = (224, 224, 3)
        self.model = _get_model(input_shape)
        self.model.save(SAVED_FLOAT_MODEL_PATH)
        print(f'Float model was saved to: {SAVED_FLOAT_MODEL_PATH}')
        self.representative_data_gen = lambda: [np.random.randn(*((1,) + input_shape))]
        # tpc = get_activation_quantization_disabled_keras_tpc('int8_test')
        self.exportable_model, _ = mct.keras_post_training_quantization_experimental(
            in_model=self.model,
            core_config=mct.CoreConfig(),
            representative_data_gen=self.representative_data_gen,
            # target_platform_capabilities=tpc,
            new_experimental_exporter=True)
        for l in self.exportable_model.layers:
            if hasattr(l, 'layer'):
                # l.use_bias = False
                if hasattr(l.layer, 'use_bias'):
                    print(f'Disabling use_bias in {l.layer.name}')
                    l.layer.use_bias=False

        tflite_export_model(model=self.exportable_model,
                            is_layer_exportable_fn=is_keras_layer_exportable,
                            mode=TFLiteExportMode.FAKELY_QUANT,
                            save_model_path=SAVED_MODEL_PATH_FQ_TFLITE)
        tflite_export_model(model=self.exportable_model,
                            is_layer_exportable_fn=is_keras_layer_exportable,
                            mode=TFLiteExportMode.INT8,
                            save_model_path=SAVED_MODEL_PATH_INT8_TFLITE)

        self.exportable_model.save(SAVED_EXPORTABLE_MODEL_PATH_TF)
        print(f'Exportable model was saved to: {SAVED_EXPORTABLE_MODEL_PATH_TF}')


    def test_compare_fq_and_int8_predictions(self):
        """
        Test that the tflite exported model can infer and that tf exported model has the same weights
        as the tflite exported model.
        """

        # Test inference of exported model
        test_image = self.representative_data_gen()[0].astype("float32")

        def _infer(inputs, model_path):
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_index = interpreter.get_input_details()[0]["index"]
            interpreter.set_tensor(input_index, inputs)
            # Run inference.
            interpreter.invoke()
            output_details = interpreter.get_output_details()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            return output_data

        fq_model_output = _infer(test_image, SAVED_MODEL_PATH_FQ_TFLITE)
        int8_model_output = _infer(test_image, SAVED_MODEL_PATH_INT8_TFLITE)
        self.assertTrue(fq_model_output.shape==int8_model_output.shape, f'Expected shapes of fq and int8 tflite models'
                                                                        f'are expected to be equal but fq output shape is {fq_model_output.shape} and int8 output shape is {int8_model_output.shape}')
        diff = fq_model_output.flatten()-int8_model_output.flatten()
        print(f'Diff shape {diff.shape}')
        maximal_error = np.max(np.abs(diff))
        self.assertTrue(np.sum(diff != 0)<=1, f'Found {np.sum(diff != 0)} different predictions')
        self.assertTrue(maximal_error==0, f'Expected outputs to be identical but max error is: {maximal_error}')

        # assert kernel_tensor_index is not None, f'did not find the kernel tensor index'

        # Test equal weights of the first conv2d layer between exported TFLite and exported TF
        # diff = np.transpose(interpreter.tensor(kernel_tensor_index)(), (1, 2, 3, 0)) - tf_exported_model.layers[2].kernel
        # self.assertTrue(np.sum(np.abs(diff)) == 0)

