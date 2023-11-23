import keras.layers
import numpy as np

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode


def prune_keras_node(node: BaseNode,
                     mask: np.ndarray,
                     fw_info: FrameworkInfo,
                     last_section_node: bool = False):

    if fw_info.is_kernel_op(node.type):
        prune_edge_linear_node(fw_info, last_section_node, mask, node)
    else:
        prune_intermediat_node(mask, node)


def prune_intermediat_node(mask, node):
    edit_node_input_shape(mask, node)
    pruned_parameters = {}
    mask_bool = mask.astype(bool)
    for k, v in node.weights.items():
        pruned_parameters[k] = v.compress(mask_bool)
    node.weights = pruned_parameters


def edit_node_input_shape(mask, node):
    new_input_shape = list(node.input_shape)
    new_input_shape[-1] = int(np.sum(mask))
    node.input_shape = tuple(new_input_shape)


def prune_edge_linear_node(fw_info, last_section_node, mask, node):
    kernel_attr = fw_info.get_kernel_op_attributes(node.type)[0]
    io_axis = fw_info.kernel_channels_mapping.get(node.type)
    axis_to_prune = io_axis[int(last_section_node)]
    kernel = node.get_weights_by_keys(kernel_attr)
    # Convert mask to boolean
    mask_bool = mask.astype(bool)
    # Prune the kernel using the mask along the specified axis
    pruned_kernel = kernel.compress(mask_bool, axis=axis_to_prune)
    node.set_weights_by_keys(name=kernel_attr, tensor=pruned_kernel)
    if node.framework_attr['use_bias'] and not last_section_node:
        bias = node.get_weights_by_keys('bias')
        pruned_bias = bias.compress(mask_bool)
        node.set_weights_by_keys(name='bias', tensor=pruned_bias)

    edit_node_attr(node, mask, last_section_node)

    if last_section_node:
        edit_node_input_shape(mask, node)


def edit_node_attr(node, mask, last_section_node):
    if node.type in [keras.layers.Conv2D, keras.layers.Conv2DTranspose]:
        if not last_section_node:
            node.framework_attr['filters'] = int(np.sum(mask))
    elif node.type == keras.layers.Dense:
        if not last_section_node:
            node.framework_attr['units'] = int(np.sum(mask))

