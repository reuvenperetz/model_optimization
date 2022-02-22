import unittest

from model_compression_toolkit.common.hardware_model.quantization_config \
    import \
    QuantizationConfigOptions

from tests.common_tests.function_tests.test_hardware_modeling import default_qc


class QCOptionsTest(unittest.TestCase):

    def test_empty_qc_options(self):
        with self.assertRaises(AssertionError) as e:
            QuantizationConfigOptions()
        self.assertEqual('Options list can not be empty', str(e.exception))

    def test_list_of_no_qc(self):
        with self.assertRaises(AssertionError) as e:
            QuantizationConfigOptions(default_qc, 3)
        self.assertEqual(
            'Options should be a list of QuantizationConfig objects, but found an object type: <class \'int\'>',
            str(e.exception))

    def test_clone_and_edit_options(self):
        options = QuantizationConfigOptions(default_qc)
        modified_options = options.clone_and_edit(activation_n_bits=3,
                                                  weights_n_bits=5)

        self.assertEqual(modified_options.quantization_config_list[0].activation_n_bits, 3)
        self.assertEqual(modified_options.quantization_config_list[0].weights_n_bits, 5)
