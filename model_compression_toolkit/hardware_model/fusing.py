#TODO: rename hw descriptor


from typing import Any

from model_compression_toolkit.common.hardware_model.hardware_model_component import \
    HardwareModelComponent


class Fusing(HardwareModelComponent):

    def __init__(self, operator_groups_list, name=None):
        super().__init__(name)
        assert isinstance(operator_groups_list,
                          list), f'List of operator groups should be a list but is {type(operator_groups_list)}'
        assert len(operator_groups_list) >= 2, f'Fusing can not be created for a single operators group'
        self.operator_groups_list = operator_groups_list

    def is_contained(self, other: Any):
        if not isinstance(other, Fusing):
            return False
        for i in range(len(self.operator_groups_list) - len(other.operator_groups_list) + 1):
            for j in range(len(other.operator_groups_list)):
                if self.operator_groups_list[i + j] != other.operator_groups_list[j]:
                    break
            else:
                return True
        return False
        # for i, o in enumerate(self.operator_groups_list):



    # def sublist(lst1, lst2):
    #     ls1 = [element for element in lst1 if element in lst2]
    #     ls2 = [element for element in lst2 if element in lst1]
    #     return ls1 == ls2

    def get_info(self):
        if self.name is not None:
            return {self.name: ' -> '.join([x.name for x in self.operator_groups_list])}
        return ' -> '.join([x.name for x in self.operator_groups_list])

