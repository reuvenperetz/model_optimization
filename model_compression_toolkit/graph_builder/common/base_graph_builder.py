from abc import ABC, abstractmethod
from typing import Any

from model_compression_toolkit.core.common import Graph


class BaseGraphBuilder(ABC):

    def build_graph(self,
                    model,
                    representative_dataset=None,
                    linear_collapsing: bool = True,
                    residual_collapsing: bool = True,
                    relu_bound_to_power_of_2=False):
        graph = self._convert_model_to_graph(model, representative_dataset)
        transformed_graph = self._transform_graph(graph, linear_collapsing, residual_collapsing)
        return transformed_graph

    @abstractmethod
    def _convert_model_to_graph(self, model: Any, representative_dataset: Any = None) -> Graph:
        raise Exception

    @abstractmethod
    def _transform_graph(self, graph: Graph,
                         linear_collapsing: bool = True,
                         residual_collapsing: bool = True) -> Graph:
        raise Exception
