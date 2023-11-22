import numpy as np

from model_compression_toolkit.core.common import BaseNode


class PruningSectionMask:
    """
    Contains the masks for the convolution nodes in a pruning section, which
    indicate which output channels remain and which are pruned.
    """

    def __init__(self,
                 first_node_input_mask: np.ndarray,
                 first_node_output_mask: np.ndarray,
                 second_node_output_mask: np.ndarray
                 ):
        """
        Initializes the PruningSectionMask with masks for the convolution nodes.

        :param conv_masks: A dictionary where keys are convolution nodes and
                           values are masks indicating the active output channels.
        """
        self.first_node_input_mask = first_node_input_mask
        self.first_node_output_mask = first_node_output_mask
        self.second_node_output_mask = second_node_output_mask


class PruningSection:
    """
    Represents a section of a neural network to be pruned, consisting of
    convolutional nodes and intermediate nodes.
    """

    def __init__(self, conv_nodes, intermediate_nodes):
        """
        Initializes the PruningSection with convolutional nodes and
        intermediate nodes.

        :param conv_nodes: A list of convolutional nodes in the section.
        :param intermediate_nodes: A list of intermediate nodes in the section.
        """
        self.conv_nodes = conv_nodes
        self.intermediate_nodes = intermediate_nodes

    def compute_memory(self, pruning_section_mask: PruningSectionMask, fw_info):
        """
        Computes the memory footprint of the section, considering the weights
        of the convolutional nodes and the size of intermediate data.
        """
        first_node_memory = get_node_num_params(node=self.conv_nodes[0],
                                            input_mask=pruning_section_mask.first_node_input_mask,
                                            output_mask=pruning_section_mask.first_node_output_mask,
                                            fw_info=fw_info)
        total_inter_nodes_memory = 0
        for inter_node in self.intermediate_nodes:
            total_inter_nodes_memory += get_node_num_params(node=inter_node,
                                                        input_mask=pruning_section_mask.first_node_output_mask,
                                                        output_mask=pruning_section_mask.first_node_output_mask,
                                                        fw_info=fw_info)

        second_node_memory = get_node_num_params(node=self.conv_nodes[1],
                                             input_mask=pruning_section_mask.first_node_output_mask,
                                             output_mask=pruning_section_mask.second_node_output_mask,
                                             fw_info=fw_info)

        return (first_node_memory + total_inter_nodes_memory + second_node_memory)*4.

    def get_pruned_section(self, pruning_section_mask):
        """
        Applies the mask to the given section, pruning the specified channels.

        :param section: The PruningSection to which the masks will be applied.
        """
        # Implement the mask application logic here
        pass


def get_node_num_params(node: BaseNode,
                        input_mask: np.ndarray,
                        output_mask: np.ndarray,
                        fw_info):
    total_params = 0
    if fw_info.is_kernel_op(node.type):
        oc_axis, ic_axis = fw_info.kernel_channels_mapping.get(node.type)
        kernel_attr = fw_info.get_kernel_op_attributes(node.type)[0]
        for w_attr, w in node.weights.items():
            if kernel_attr in w_attr:
                # Convert masks to boolean arrays, treating None as all ones
                input_mask = np.ones(w.shape[ic_axis], dtype=bool) if input_mask is None else input_mask.astype(bool)
                output_mask = np.ones(w.shape[oc_axis], dtype=bool) if output_mask is None else output_mask.astype(bool)

                # Apply masks to the kernel using np.take
                pruned_w = np.take(w, np.where(input_mask)[0], axis=ic_axis)
                pruned_w = np.take(pruned_w, np.where(output_mask)[0], axis=oc_axis)

                # Calculate the number of remaining parameters
                total_params += len(pruned_w.flatten())
            else:
                total_params += len(np.take(w, np.where(output_mask)[0]))

    else:
        for w_attr, w in node.weights.items():
            total_params += len(np.take(w, np.where(output_mask)[0]))

    return total_params
