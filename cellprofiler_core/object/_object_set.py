from ._objects import Objects


class ObjectSet:
    """A set of objects.Objects instances.

    This class allows you to either refer to an object by name or
    iterate over all available objects.
    """

    def __init__(self, can_overwrite=False):
        """Initialize the object set

        can_overwrite - True to allow overwriting of a new copy of objects
                        over an old one of the same name (for debugging)
        """
        self.__can_overwrite = can_overwrite
        self.__types_and_instances = {"objects": {}}

    @property
    def __objects_by_name(self):
        return self.__types_and_instances["objects"]

    def add_objects(self, objects, name):
        assert isinstance(
            objects, Objects
        ), "objects must be an instance of CellProfiler.Objects"
        assert (
            name not in list(self.__objects_by_name.keys())
        ) or self.__can_overwrite, (
            "The object, {}, is already in the object set".format(name)
        )
        self.__objects_by_name[name] = objects

    def get_object_names(self):
        """Return the names of all of the objects
        """
        return list(self.__objects_by_name.keys())

    object_names = property(get_object_names)

    def get_objects(self, name):
        """Return the objects instance with the given name
        """
        return self.__objects_by_name[name]

    @property
    def all_objects(self):
        """Return a list of name / objects tuples
        """
        return list(self.__objects_by_name.items())

    def get_types(self):
        """Get then names of types of per-image set "things"

        The object set can store arbitrary types of things other than objects,
        for instance ImageJ data tables. This function returns the thing types
        defined in the object set at this stage of the pipeline.
        """
        return list(self.__types_and_instances.keys())

    def add_type_instance(self, type_name, instance_name, instance):
        """Add a named instance of a type

        A thing of a given type can be stored in the object set so that
        it can be retrieved by name later in the pipeline. This function adds
        an instance of a type to the object set.

        type_name - the name of the instance's type
        instance_name - the name of the instance
        instance - the instance itself
        """
        if type_name not in self.__types_and_instances:
            self.__types_and_instances[type_name] = {}
        self.__types_and_instances[type_name][instance_name] = instance

    def get_type_instance(self, type_name, instance_name):
        """Get an named instance of a type

        type_name - the name of the type of instance
        instance_name - the name of the instance to retrieve
        """
        if (
            type_name not in self.__types_and_instance
            or instance_name not in self.__types_and_instances[type_name]
        ):
            return None
        return self.__types_and_instances[type_name][instance_name]
