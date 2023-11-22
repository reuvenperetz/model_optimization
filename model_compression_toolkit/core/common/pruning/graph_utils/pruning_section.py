
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
        first_node_memory = MemoryHelper.get_node_num_params(node=self.conv_nodes[0],
                                                             input_mask=pruning_section_mask.first_node_input_mask,
                                                             output_mask=pruning_section_mask.first_node_output_mask,
                                                             fw_info=fw_info)
        total_inter_nodes_memory = 0
        for inter_node in self.intermediate_nodes:
            total_inter_nodes_memory += MemoryHelper.get_node_num_params(node=inter_node,
                                                                         input_mask=pruning_section_mask.first_node_output_mask,
                                                                         output_mask=pruning_section_mask.first_node_output_mask,
                                                                         fw_info=fw_info)

        second_node_memory = MemoryHelper.get_node_num_params(node=self.conv_nodes[1],
                                                              input_mask=pruning_section_mask.first_node_output_mask,
                                                              output_mask=pruning_section_mask.second_node_output_mask,
                                                              fw_info=fw_info)

        return (first_node_memory + total_inter_nodes_memory + second_node_memory)*4.


