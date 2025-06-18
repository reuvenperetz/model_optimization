from dataclasses import dataclass

@dataclass
class GraphRefinementConfig:
    fold_batch_norm: bool = True
    collapse_linear_layers: bool = True
    remove_identity_nodes: bool = True
