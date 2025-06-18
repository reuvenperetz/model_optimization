from dataclasses import dataclass
from typing import Optional, Dict
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.graph_builder.common.graph_refinement_config import GraphRefinementConfig


@dataclass
class KerasGraphRefinementConfig(GraphRefinementConfig):
    separable_decomposition: bool = True


# TODO: change transform
def refine_keras_graph(
    graph: Graph,
    graph_refinement_config: KerasGraphRefinementConfig = None
) -> Graph:
    """
    Applies a series of structural simplifications to a graph.

    This includes transformations such as batch normalization folding, merging linear layers, etc.
    These transformations are aimed at simplifying the graph for optimization without altering the model's functionality.

    Args:
        graph (Graph): The input graph to refine.
        graph_refinement_config (Optional[Dict[str, bool]]): An optional dictionary to enable/disable specific refinements.

    Returns:
        Graph: A refined graph with structural transformations applied.

    Notes:
        This function does not perform numerical optimizations (e.g., quantization),
        nor does it alter weights or model accuracy. It is purely structural.
    """
    pass
