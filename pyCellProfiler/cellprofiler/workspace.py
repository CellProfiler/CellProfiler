"""workspace.py - the workspace for an imageset

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import cellprofiler.gui.cpfigure as cpf

class Workspace(object):
    """The workspace contains the processing information and state for
    a pipeline run on an image set
    """
    def __init__(self,
                 pipeline,
                 module,
                 image_set,
                 object_set,
                 measurements,
                 image_set_list,
                 frame=None,
                 create_new_window = False):
        """Workspace constructor
        
        pipeline          - the pipeline of modules being run
        module            - the current module to run
        image_set         - the set of images available for this iteration
        object_set        - an object.ObjectSet instance
        image_set_list    - the list of all images
        frame             - the application's frame, or None for no display
        create_new_window - True to create another frame, even if one is open
                            False to reuse the current frame.
        """
        self.__pipeline = pipeline
        self.__module = module
        self.__image_set = image_set
        self.__object_set = object_set
        self.__measurements = measurements
        self.__image_set_list = image_set_list
        self.__frame = frame
        self.__outlines = {}
        self.__windows_used = []
        self.__create_new_window = create_new_window
    
    def refresh(self):
        """Refresh any windows created during use"""
        for window in self.__windows_used:
            window.Refresh()
    
    def get_windows_used(self):
        return self.__windows_used

    def get_pipeline(self):
        """Get the pipeline being run"""
        return self.__pipeline
    pipeline = property(get_pipeline)
    
    def get_image_set(self):
        """The image set is the set of images currently being processed
        """
        return self.__image_set
    image_set = property(get_image_set)
    
    def get_image_set_list(self):
        """The list of all image sets"""
        return self.__image_set_list
    image_set_list = property(get_image_set_list)

    def get_object_set(self):
        """The object set is the set of image labels for the current image set
        """
        return self.__object_set
    
    object_set = property(get_object_set)

    def get_objects(self,objects_name):
        """Return the objects.Objects instance for the given name.
        
        objects_name - the name of the objects to retrieve
        """
        return self.object_set.get_objects(objects_name)

    def get_measurements(self):
        """The measurements contain measurements made on images and objects
        """
        return self.__measurements

    measurements = property(get_measurements)
    
    def add_measurement(self, object_name, feature_name, data):
        """Add a measurement to the workspace's measurements
        
        object_name - name of the objects measured or 'Image'
        feature_name - name of the feature measured
        data - the result of the measurement
        """
        self.measurements.add_measurement(object_name, feature_name, data)

    def get_frame(self):
        """The frame is CellProfiler's gui window

        If the frame is present, a module should do its display
        """
        return self.__frame

    frame = property(get_frame)
    
    def get_display(self):
        """True to provide a gui display"""
        return self.__frame != None
    display = property(get_display)
    
    def create_or_find_figure(self,title=None,subplots=None,window_name = None):
        """Create a matplotlib figure window or find one already created"""
        if title==None:
            title=self.__module.module_name
            
        if window_name == None:
            window_name = "CellProfiler:%s:%s"%(self.__module.module_name,
                                                self.__module.module_num)
        if self.__create_new_window:
            figure = CPFigureFrame(self, 
                                   title=title,
                                   name = window_name,
                                   subplots = subplots)
        else:
            figure = cpf.create_or_find(self.__frame, title = title, 
                                        name = window_name, 
                                        subplots = subplots)
        if not figure in self.__windows_used:
            self.__windows_used.append(figure)
        return figure
    
    def get_outline_names(self):
        """The names of outlines of objects"""
        return self.__outlines.keys()
    
    def add_outline(self, name, outline):
        """Add an object outline to the workspace"""
        self.__outlines[name] = outline
    
    def get_outline(self, name):
        """Get a named outline"""
        return self.__outlines[name]
    
    def get_module(self):
        """Get the module currently being run"""
        return self.__module
    
    module = property(get_module)
    
    @property
    def is_last_image_set(self):
        return (self.measurements.image_set_number ==
                self.image_set_list.count()-1)
        
