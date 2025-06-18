from typing import Iterable, TYPE_CHECKING

from model_compression_toolkit.core.common import Graph

if TYPE_CHECKING:
    import torch

def convert_pytorch_model_to_graph(model: "torch.nn.Module",
                                   representative_dataset: Iterable) -> Graph:
    """
    Converts a PyTorch model into a computational graph using tracing.

    This function requires a representative dataset to trace the dynamic
    execution of the model and build a corresponding graph.

    Args:
        model (torch.nn.Module): The PyTorch model to convert.
        representative_dataset (Iterable): An iterable yielding sample inputs for tracing the model.

    Returns:
        Graph: A graph containing nodes and edges representing the model.

    """
    pass
