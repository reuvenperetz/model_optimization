# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Dict
from model_compression_toolkit.constants import MCT_VERSION, TPC_VERSION

def create_model_metadata(tpc, scheduling_info = None) -> Dict:
    _metadata = get_versions_dict(tpc)
    if scheduling_info:
        scheduler_metadata = get_scheduler_metadata(scheduler_info=scheduling_info)
        _metadata['scheduling_info'] = scheduler_metadata
    return _metadata



def get_versions_dict(tpc) -> Dict:
    """

    Returns: A dictionary with TPC and MCT versions.

    """
    # imported inside to avoid circular import error
    from model_compression_toolkit import __version__ as mct_version
    tpc_version = f'{tpc.name}.{tpc.version}'
    return {MCT_VERSION: mct_version, TPC_VERSION: tpc_version}

def get_scheduler_metadata(scheduler_info):
    def serialize_cut(cut):
        return {
            'op_order': [op.name for op in cut.op_order],
            'op_record': [op.name for op in cut.op_record],
            'mem_elements': [
                {
                    'shape': tensor.shape,
                    'node_name': tensor.node_name,
                    'total_size': tensor.total_size,
                    'node_output_index': tensor.node_output_index
                }
                for tensor in cut.mem_elements.elements
            ]
        }

    serialized_info = {
        'schedule': [str(layer) for layer in scheduler_info['schedule']],
        'max_cut': scheduler_info['max_cut'],
        'cuts': [serialize_cut(cut) for cut in scheduler_info['cuts']],
        'fused_nodes_mapping': scheduler_info['fused_nodes_mapping']
    }

    return serialized_info
