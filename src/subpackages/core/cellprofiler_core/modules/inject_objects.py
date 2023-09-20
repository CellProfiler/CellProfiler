from ..module import Module
from ..object import Objects
from ..setting.text.alphanumeric.name._label_name import LabelName


class InjectObjects(Module):
    """Inject objects with labels into the pipeline"""

    module_name = "InjectObjects"
    variable_revision_number = 1

    def __init__(
        self,
        object_name,
        segmented,
        unedited_segmented=None,
        small_removed_segmented=None,
    ):
        """Initialize the module with the objects for the object set

        object_name - name of the objects to be provided
        segmented   - labels for the segmentation of the image
        unedited_segmented - labels including small and boundary, default =
                             same as segmented
        small_removed_segmented - labels with small objects removed, default =
                                  same as segmented
        """
        super(InjectObjects, self).__init__()
        self.object_name = LabelName("text", object_name)
        self.__segmented = segmented
        self.__unedited_segmented = unedited_segmented
        self.__small_removed_segmented = small_removed_segmented

    def settings(self):
        return [self.object_name]

    def run(self, workspace):
        my_objects = Objects()
        my_objects.segmented = self.__segmented
        if self.__unedited_segmented is not None:
            my_objects.unedited_segmented = self.__unedited_segmented
        if self.__small_removed_segmented is not None:
            my_objects.small_removed_segmented = self.__small_removed_segmented
        workspace.object_set.add_objects(my_objects, self.object_name.value)
