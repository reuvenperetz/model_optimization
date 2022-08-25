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
import numpy as np
import random
import unittest
from torchvision.models import mobilenet_v2

import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor

tp = mct.target_platform

import torch


class TestPytorchExport(unittest.TestCase):

    def test_sanity(self):
        repr_dataset = lambda: to_torch_tensor([torch.randn(1, 3, 224, 224)])
        core_config = mct.CoreConfig(n_iter=1)
        model, _ = mct.pytorch_post_training_quantization_experimental(
            in_module=mobilenet_v2(pretrained=True),
            representative_data_gen=repr_dataset,
            core_config=core_config)
        images = repr_dataset()
        model(images)
