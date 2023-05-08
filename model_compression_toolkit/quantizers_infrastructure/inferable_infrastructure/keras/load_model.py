# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import FOUND_TF
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.get_all_subclasses import get_all_subclasses

if FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit import quantizers_infrastructure as qi
    from model_compression_toolkit.quantizers_infrastructure import BaseKerasTrainableQuantizer
    from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizers.base_keras_inferable_quantizer import BaseKerasInferableQuantizer
    keras = tf.keras

    def keras_load_quantized_model(filepath, custom_objects=None, compile=True, options=None):
        """
        This function wraps the keras load model and MCT quantization custom class to it.

        Args:
            filepath: the model file path.
            custom_objects: Additional custom objects
            compile: Boolean, whether to compile the model after loading.
            options: Optional `tf.saved_model.LoadOptions` object that specifies options for loading from SavedModel.

        Returns: A keras Model

        """
        qi_inferable_custom_objects = {subclass.__name__: subclass for subclass in
                                       get_all_subclasses(BaseKerasInferableQuantizer)}
        all_inferable_names = list(qi_inferable_custom_objects.keys())
        if len(set(all_inferable_names)) < len(all_inferable_names):
            Logger.error(f"Found multiple quantizers with the same name that inherit from BaseKerasInferableQuantizer"
                         f"while trying to load a model.")

        qi_trainable_custom_objects = {subclass.__name__: subclass for subclass in
                                       get_all_subclasses(BaseKerasTrainableQuantizer)}
        all_trainable_names = list(qi_trainable_custom_objects.keys())
        if len(set(all_trainable_names)) < len(all_trainable_names):
            Logger.error(f"Found multiple quantizers with the same name that inherit from BaseKerasTrainableQuantizer"
                         f"while trying to load a model.")

        # Merge dictionaries into one dict
        qi_custom_objects = {**qi_inferable_custom_objects, **qi_trainable_custom_objects}

        # Add non-quantizers custom objects
        qi_custom_objects.update({qi.KerasQuantizationWrapper.__name__: qi.KerasQuantizationWrapper})
        qi_custom_objects.update({qi.ActivationQuantizationHolder.__name__: qi.ActivationQuantizationHolder})

        if custom_objects is not None:
            qi_custom_objects.update(custom_objects)
        return tf.keras.models.load_model(filepath,
                                          custom_objects=qi_custom_objects, compile=compile,
                                          options=options)
else:
    def keras_load_quantized_model(filepath, custom_objects=None, compile=True, options=None):
        """
        This function wraps the keras load model and MCT quantization custom class to it.

        Args:
            filepath: the model file path.
            custom_objects: Additional custom objects
            compile: Boolean, whether to compile the model after loading.
            options: Optional `tf.saved_model.LoadOptions` object that specifies options for loading from SavedModel.

        Returns: A keras Model

        """
        Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                        'when using keras_load_quantized_model. '
                        'Could not find Tensorflow package.')  # pragma: no cover
