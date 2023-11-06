import numpy as np

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode


def prune_keras_node(node: BaseNode,
                     mask: np.ndarray,
                     fw_info: FrameworkInfo,
                     prune_input_channels: bool = False):

    if fw_info.is_kernel_op(node.type):
        kernel_attr = fw_info.get_kernel_op_attributes(node.type)[0]
        if prune_input_channels:
            axis_to_prune = fw_info.kernel_channels_mapping.get(node.type)[1]
        else:
            axis_to_prune = fw_info.kernel_channels_mapping.get(node.type)[0]

        kernel = node.get_weights_by_keys(kernel_attr)
        # Convert mask to boolean
        mask_bool = mask.astype(bool)
        # try:
        # Prune the kernel using the mask along the specified axis
        pruned_kernel = kernel.compress(mask_bool, axis=axis_to_prune)
        # except Exception as e:
        #     print(e)
        node.set_weights_by_keys(name=kernel_attr, tensor=pruned_kernel)
        if node.framework_attr['use_bias'] and not prune_input_channels:
            bias = node.get_weights_by_keys('bias')
            # try:
            pruned_bias = bias.compress(mask_bool)
            # except Exception as e:
            #     print(e)
            node.set_weights_by_keys(name='bias', tensor=pruned_bias)


        if not prune_input_channels:
            node.framework_attr['filters']=int(np.sum(mask))
        else:
            new_input_shape = list(node.input_shape)
            new_input_shape[-1] = int(np.sum(mask))
            node.input_shape = tuple(new_input_shape)
    else:
        new_input_shape = list(node.input_shape)
        new_input_shape[-1] = int(np.sum(mask))
        node.input_shape = tuple(new_input_shape)
        pruned_parameters = {}
        mask_bool = mask.astype(bool)
        for k,v in node.weights.items():
            pruned_parameters[k] = v.compress(mask_bool)

        node.weights = pruned_parameters





