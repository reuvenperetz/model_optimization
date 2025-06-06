:orphan:

.. _ug-GradientPTQConfig:


=================================
GradientPTQConfig Class
=================================


**The following API can be used to create a GradientPTQConfig instance which can be used for post training quantization using knowledge distillation from a teacher (float model) to a student (the quantized model)**

.. autoclass:: model_compression_toolkit.gptq.GradientPTQConfig
    :members:

=================================
GPTQHessianScoresConfig Class
=================================


**The following API can be used to create a GPTQHessianScoresConfig instance which can be used to define necessary parameters for computing Hessian scores for the GPTQ loss function.**

.. autoclass:: model_compression_toolkit.gptq.GPTQHessianScoresConfig
    :members:


=================================
RoundingType
=================================

.. autoclass:: model_compression_toolkit.gptq.RoundingType
    :members:


=====================================
GradualActivationQuantizationConfig
=====================================

**The following API can be used to configure the gradual activation quantization when using GPTQ.**

.. autoclass:: model_compression_toolkit.gptq.GradualActivationQuantizationConfig
    :members:


=====================================
QFractionLinearAnnealingConfig
=====================================

.. autoclass:: model_compression_toolkit.gptq.QFractionLinearAnnealingConfig
    :members:

