from typing import Optional, Dict
from model_compression_toolkit.core.common import Graph
from dataclasses import dataclass

from model_compression_toolkit.graph_builder.common.graph_refinement_config import GraphRefinementConfig


def transform_keras_graph(graph: Graph,
                          linear_collapsing: bool = True,
                          residual_collapsing: bool = True) -> Graph:
    """
    Applies a series of structural simplifications to a graph.

    This includes transformations such as batch normalization folding, merging linear layers, etc.
    These transformations are aimed at simplifying the graph for optimization without altering the model's functionality.

    Args:
        graph (Graph): The input graph to transform.
        linear_collapsing:
        residual_collapsing:

    Returns:
        Graph: A refined graph with structural transformations applied.

    Notes:
        This function does not perform numerical optimizations (e.g., quantization),
        nor does it alter weights or model accuracy. It is purely structural.
    """
    pass
