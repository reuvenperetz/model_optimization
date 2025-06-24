from typing import Any

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.graph_builder import convert_keras_model_to_graph
from model_compression_toolkit.graph_builder.common.base_graph_builder import BaseGraphBuilder
from model_compression_toolkit.graph_builder.keras.transform_keras_graph import transform_keras_graph


class KerasGraphBuilder(BaseGraphBuilder):
    def convert_model_to_graph(self, model: Any, representative_dataset: Any = None) -> Graph:
        return convert_keras_model_to_graph(model)

    def transform_graph(self,
                        graph: Graph,
                        linear_collapsing: bool = True,
                        residual_collapsing: bool = True
                        ) -> Graph:
        return transform_keras_graph(graph,
                                     linear_collapsing,
                                     residual_collapsing)