from typing import TYPE_CHECKING
from model_compression_toolkit.core.common import Graph


if TYPE_CHECKING:
    import tensorflow as tf

#TODO: wrap the entire graph building logic in an abstract class that each framewrok implement
#TODO: maybe just skip the configuration and do it all hard-coded except for the flags the user has access to
def convert_keras_model_to_graph(model: "tf.keras.Model") -> Graph:
    """
    Converts a Keras model into a computational graph.

    This function analyzes the structure of a Keras model
    and builds a graph representation from its layers and connections.

    Args:
        model (tf.keras.Model): The Keras model to convert.

    Returns:
        ComputationalGraph: A graph containing nodes and edges representing the model.
    """
    pass
