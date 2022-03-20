# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

class CurrentHardwareModel:
    """A thread-local stack of objects for providing implicit defaults."""

    def __init__(self):
        super(CurrentHardwareModel, self).__init__()
        self.hwm = None

    def get(self):
        if self.hwm is None:
            raise Exception('Hardware model is not initialized.')
        return self.hwm

    def reset(self):
        self.hwm = None

    def set(self, hwm):
        self.hwm = hwm


_current_hardware_model = CurrentHardwareModel()

def get_current_model():
    return _current_hardware_model.get()
