"""workspace.py - the workspace for an imageset
"""
__version__="$Revision: 1 "

class Workspace(object):
    """The workspace contains the processing information and state for
    a pipeline run on an image set
    """
    def __init__(self,module,image_set,object_set,measurements,frame=None):
        self.__module = module
        self.__image_set = image_set
        self.__object_set = object_set
        self.__measurements = measurements
        self.__frame = frame

    def get_image_set(self):
        """The image set is the set of images currently being processed
        """
        return self.__image_set
    image_set = property(get_image_set)

    def get_object_set(self):
        """The object set is the set of image labels for the current image set
        """
        return self.__object_set

    object_set = property(get_object_set)

    def get_measurements(self):
        """The measurements contain measurements made on images and objects
        """
        return self.__measurements

    measurements = property(get_measurements)

    def get_frame(self):
        """The frame is CellProfiler's gui window

        If the frame is present, a module should do its display
        """
        return self.__frame

    frame = property(get_frame)
    
    def create_or_find_figure(self,title=None,subplots=None):
        pass
