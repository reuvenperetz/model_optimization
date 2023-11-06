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


import unittest

import tensorflow as tf

from tests.keras_tests.pruning_tests.networks_tests.conv2d_pruning_test import Conv2DPruningTest


class PruningNetworksTest(unittest.TestCase):

    def test_conv2d_pruning(self):
        Conv2DPruningTest(self).run_test()


if __name__ == '__main__':
    unittest.main()
