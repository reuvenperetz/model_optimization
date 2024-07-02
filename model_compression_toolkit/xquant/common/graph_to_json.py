#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import json

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode
import torch


class GraphToJsonConverter:
    def __init__(self, graph: Graph, similarity_metrics:dict):
        self.graph = graph
        self.similarity_metrics = similarity_metrics

    def convert(self):
        node_name_to_node_dict = {}
        for node in self.graph.nodes:
            node_dict = {
                "id": node.name,
                "op": node.type.__name__,
                "properties": {
                    "attributes": flatten_and_convert_dict(node.framework_attr),
                },
                "inputs": [],
                "outputs": []
            }
            node_name_to_node_dict[node.name] = node_dict

        for node in self.graph.nodes:
            _incoming_edges = self.graph.incoming_edges(node)
            for _in_edge in _incoming_edges:
                input_dict = {
                    "index": _in_edge.sink_index,
                    "isConst": False,
                    "sourceNodeId": _in_edge.source_node.name,
                    "sourceOutputIndex": _in_edge.source_index,
                    "shape": _in_edge.source_node.output_shape[_in_edge.source_index]
                }
                node_name_to_node_dict[node.name]['inputs'].append(input_dict)

            _output_shapes = node.output_shape
            # _outgoing_edged = self.graph.out_edges(node)
            for _index, _out_shape in enumerate(_output_shapes):
                output_dict = {"index": _index,
                               "shape": _out_shape}
                node_name_to_node_dict[node.name]['outputs'].append(output_dict)

        for input_node in self.graph.get_inputs():
            node_name_to_node_dict[input_node.name]['op'] = 'Placeholder'

        metadata = {
            "conversionStatus": "SUCCEEDED",
            "converterVersion": "5.0.4",
            "created": "Tue Jun 18 09:44:57 UTC 2024",
            "memory": {
                "staticMemory": 9472,
                "runtimeMemory": 72704,
                "totalRequiredMemory": 83200,
                "availableMemory": 8388480
            }
        }

        insights = self.generate_insights()

        json_dict = {
            "nodes": list(node_name_to_node_dict.values()),
            "metadata": metadata,
            "insights": [i.to_dict() for i in insights]
        }

        return json_dict

    def generate_insights(self):
        insights = []
        insights.append(Insight(name="Models outputs similarity",
                                category='Outputs Similarity',
                                description='Similarity metrics between outputs of the two models',
                                type="Text",
                                severity='Warning',
                                insight=SimilarityMetricsFormatter(self.similarity_metrics).format_metrics()))
        snc_insight = self.get_snc_insight()
        if snc_insight:
            insights.append(snc_insight)
        residual_collapse_insight = self.get_linear_collapsing_insight()
        if residual_collapse_insight:
            insights.append(residual_collapse_insight)
        _collapse_insight = self.get_residual_collapsing_insight()
        if _collapse_insight:
            insights.append(_collapse_insight)

        return insights

    def get_snc_insight(self):
        suspected_nodes = []
        for node in self.graph.nodes:
            if node.type in [torch.nn.modules.activation.SiLU]:
                silu_next_nodes = self.graph.get_next_nodes(node)
                assert silu_next_nodes[0].type==torch.fake_quantize_per_tensor_affine, f'something weird. silu is not quantized, check me'
                silu_next_next_nodes = self.graph.get_next_nodes(silu_next_nodes[0])
                if silu_next_next_nodes[0].type in [torch.nn.Conv2d]: # SNC can help here
                    suspected_nodes.append(node.name)
        if len(suspected_nodes)==0:
            return None
        subgraph={"nodes":[]}
        for suspected_node in suspected_nodes:
            subgraph["nodes"].append({"id": suspected_node})
        insight = Insight(name="SNC Usage",
                          category="MCT Features Usage",
                          description="Shift Negative Correction was not used or misused",
                          type="Graph",
                          severity="Warning",
                          insight="Shift Negative Correction (SNC) can enhance performance in models with non-linear layers featuring negative activations, such as Swish and PReLU. However, some layers may not receive this optimization, potentially due to SNC being disabled or incorrect SNC parameter usage. For further information, please refer to this page: https://some.page.that.explains.about.snc.and.how.to.use.it.",
                          subgraph=subgraph
                          )
        return insight

    def get_linear_collapsing_insight(self):
        suspected_nodes = []
        for node in self.graph.get_topo_sorted_nodes():
            if node.type in [torch.nn.Conv2d] and len(self.graph.get_next_nodes(node))==1 and len(self.graph.get_prev_nodes(node))==1:
                conv_next_nodes = self.graph.get_next_nodes(node)
                conv_next_next_nodes = self.graph.get_next_nodes(conv_next_nodes[0])
                if len(conv_next_next_nodes)>0:
                    if conv_next_next_nodes[0].type in [torch.nn.Conv2d]:
                        suspected_nodes.append(node.name)
        print(suspected_nodes)
        if len(suspected_nodes)==0:
            return None
        subgraph={"nodes":[]}
        for suspected_node in suspected_nodes:
            subgraph["nodes"].append({"id": suspected_node})
        insight = Insight(name="Linear Collapsing Usage",
                          category="MCT Features Usage",
                          description="Linear Collapsing was not used",
                          type="Graph",
                          severity="Warning",
                          insight="Linear collapsing merge adjacent linear layers and help reduce memory, but it seems it was not used.\nTo use it please set linear_collapins to True in mct.core.QuantizationConfig.\nFor more details, please visit our API here: https://sony.github.io/model_optimization/docs/api/api_docs/classes/QuantizationConfig.html",
                          subgraph=subgraph
                          )
        return insight

    def get_residual_collapsing_insight(self):
        suspected_nodes = []
        for node in self.graph.get_topo_sorted_nodes():
            if node.type in [torch.nn.Conv2d]:
                conv_next_nodes = self.graph.get_next_nodes(node)
                conv_next_next_nodes = self.graph.get_next_nodes(conv_next_nodes[0])
                if len(conv_next_next_nodes)>0:
                    if conv_next_next_nodes[0].name=='add':
                        suspected_nodes.append(node.name)
                        suspected_nodes.append(conv_next_next_nodes[0].name)
        print(suspected_nodes)
        if len(suspected_nodes)==0:
            return None
        subgraph={"nodes":[]}
        for suspected_node in suspected_nodes:
            subgraph["nodes"].append({"id": suspected_node})
        insight = Insight(name="Residual Collapsing Usage",
                          category="MCT Features Usage",
                          description="Residual Collapsing was not used",
                          type="Graph",
                          severity="Warning",
                          insight="Residual collapsing allow removing residual connections of linear layers and help reduce memory and latency, but it seems it was not used.\nTo use it please set residual_collapins to True in mct.core.QuantizationConfig.\nFor more details, please visit our API here: https://sony.github.io/model_optimization/docs/api/api_docs/classes/QuantizationConfig.html",
                          subgraph=subgraph
                          )
        return insight

def flatten_and_convert_dict(d):
    def convert_key(key):
        parts = key.split('_')
        return parts[0] + ''.join(word.capitalize() for word in parts[1:])

    def flatten_dict(d, parent_key='', sep=''):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{convert_key(k)}" if parent_key else convert_key(k)
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep))
            else:
                items[new_key] = str(v)
        return items

    return flatten_dict(d)


class Insight:
    def __init__(self, name, category, description, type, severity, insight, subgraph=None):
        self.name = name
        self.category = category
        self.description = description
        self.type = type
        self.severity = severity
        self.subgraph = subgraph
        self.insight = insight

    def to_dict(self):
        _dict= {'name': self.name,
                'category': self.category,
                'description': self.description,
                'type': self.type,
                'severity': self.severity,
                'insight': self.insight
                }
        if self.subgraph:
            _dict.update({'subgraph': self.subgraph})
        return _dict

class SimilarityMetricsFormatter:
    def __init__(self, similarity_metrics):
        self.similarity_metrics = similarity_metrics

    def format_metrics(self):
        output_repr = self.similarity_metrics.get('output_similarity_metrics_repr', {})
        output_val = self.similarity_metrics.get('output_similarity_metrics_val', {})

        repr_metrics = ', '.join([f"{key.upper()}: {value:.2f}" for key, value in output_repr.items()])
        val_metrics = ', '.join([f"{key.upper()}: {value:.2f}" for key, value in output_val.items()])

        formatted_string = f"Output Similarity Metrics Repr: \n{repr_metrics}\nOutput Similarity Metrics Val: \n{val_metrics}"
        return formatted_string