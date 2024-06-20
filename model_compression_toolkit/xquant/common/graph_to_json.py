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


class GraphToJsonConverter:
    def __init__(self, graph: Graph):
        self.graph = graph

    def convert(self):
        nodes = []
        node: BaseNode
        for node in self.graph.nodes:
            node_dict = {
                "id": node.name,
                "op": node.type,
                "properties": {
                    "attributes": node.framework_attr,
                },
                "inputs": [],
                "outputs": []
            }
            nodes.append(node_dict)

        for node in self.graph.nodes:
            _incoming_nodes = self.graph.get_prev_nodes(node)
            _incoming_edges = self.graph.incoming_edges(node)

            for _in_edge in _incoming_edges:
                input_dict = {
                    "index": _in_edge.sink_index,
                    "isConst": False,
                    "sourceNodeId": _in_edge.source_node.name,
                    "sourceOutputIndex": _in_edge.source_index,
                    "shape": _in_edge.source_node.output_shape[_in_edge.source_index]
                }


            for i, (source, target, key, edge_data) in enumerate(self.graph.edges(node, data=True, keys=True)):
                if "input" in edge_data:
                    input_dict = {
                        "index": i,
                        "isConst": False,
                        "sourceNodeId": source,
                        "sourceOutputIndex": edge_data.get("sourceOutputIndex", 0),
                        "shape": edge_data.get("shape", [])
                    }
                    next(node for node in nodes if node["id"] == target)["inputs"].append(input_dict)
                if "output" in edge_data:
                    output_dict = {
                        "index": i,
                        "shape": edge_data.get("shape", [])
                    }
                    next(node for node in nodes if node["id"] == source)["outputs"].append(output_dict)

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

        json_dict = {
            "nodes": nodes,
            "metadata": metadata
        }

        return json.dumps(json_dict, indent=4)


