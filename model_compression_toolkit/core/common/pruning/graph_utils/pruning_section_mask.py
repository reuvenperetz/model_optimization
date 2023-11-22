import numpy as np
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
        self.first_node_input_mask = first_node_input_mask
        self.first_node_output_mask = first_node_output_mask
        self.second_node_output_mask = second_node_output_mask

