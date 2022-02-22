from model_compression_toolkit import FrameworkInfo, FrameworkHardwareModel
from model_compression_toolkit.common import Graph
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation


class OptimizationContext:
    def __init__(self,
                 graph: Graph,
                 fw_info: FrameworkInfo,
                 fw_impl: FrameworkImplementation,
                 fw_hw_model: FrameworkHardwareModel):
        """

        Args:
            graph:
            fw_info:
            fw_impl:
            fw_hw_model:
        """
        self.graph = graph
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.fw_hw_model = fw_hw_model
