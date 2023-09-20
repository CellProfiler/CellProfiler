import functools

from ._module import Module
from ..constants.measurement import C_CHILDREN
from ..constants.measurement import C_COUNT
from ..constants.measurement import C_LOCATION
from ..constants.measurement import C_NUMBER
from ..constants.measurement import C_PARENT
from ..constants.measurement import FTR_CENTER_X
from ..constants.measurement import FTR_CENTER_Y
from ..constants.measurement import FTR_OBJECT_NUMBER
from ..constants.measurement import IMAGE


class Identify(Module):
    @staticmethod
    def get_object_categories(pipeline, object_name, object_dictionary):
        """Get categories related to creating new children

        pipeline - the pipeline being run (not used)
        object_name - the base object of the measurement: "Image" or an object
        object_dictionary - a dictionary where each key is the name of
                            an object created by this module and each
                            value is a list of names of parents.
        """
        if object_name == IMAGE:
            return [C_COUNT]
        result = []
        if object_name in object_dictionary:
            result += [
                C_LOCATION,
                C_NUMBER,
            ]
            if len(object_dictionary[object_name]) > 0:
                result += [C_PARENT]
        if object_name in functools.reduce(
            lambda x, y: x + y, list(object_dictionary.values())
        ):
            result += [C_CHILDREN]
        return result

    @staticmethod
    def get_object_measurements(pipleline, object_name, category, object_dictionary):
        """Get measurements related to creating new children

        pipeline - the pipeline being run (not used)
        object_name - the base object of the measurement: "Image" or an object
        object_dictionary - a dictionary where each key is the name of
                            an object created by this module and each
                            value is a list of names of parents.
        """
        if object_name == IMAGE and category == C_COUNT:
            return list(object_dictionary.keys())

        if object_name in object_dictionary:
            if category == C_LOCATION:
                return [
                    FTR_CENTER_X,
                    FTR_CENTER_Y,
                ]
            elif category == C_NUMBER:
                return [FTR_OBJECT_NUMBER]
            elif category == C_PARENT:
                return list(object_dictionary[object_name])
        if category == C_CHILDREN:
            result = []
            for child_object_name in list(object_dictionary.keys()):
                if object_name in object_dictionary[child_object_name]:
                    result += ["%s_Count" % child_object_name]
            return result
        return []
