"""
InjectImage.py - for testing, this module injects a single image into the image set

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import cellprofiler.cpmodule
import cellprofiler.cpimage
import cellprofiler.settings
import cellprofiler.objects

class InjectImage(cellprofiler.cpmodule.CPModule):
    """This module is intended for testing. It injects an image into the
    image set.
    """
    module_name = "InjectImage"
    variable_revision_number = 1

    def __init__(self, image_name, image, mask=None, release_image = False):
        super(InjectImage,self).__init__()
        self.__image_name = image_name
        self.__image = image
        self.__mask  = mask
        self.image_name = cellprofiler.settings.NameProvider("Hardwired image name","imagegroup",image_name) 
        self.__release_image = release_image
    
    def settings(self):
        return [self.image_name]
    
    def visible_settings(self):
        return [self.image_name]
    
    def get_help(self):
        """Return help text for the module
        
        """
        raise NotImplementedError("Please implement GetHelp in your derived module class")
            
    variable_revision_number = 1
    
    def write_to_handles(self,handles):
        """Write out the module's state to the handles
        
        """
    
    def write_to_text(self,file):
        """Write the module's state, informally, to a text file
        """
    
    def prepare_run(self, workspace):
        workspace.image_set_list.get_image_set(0)
        return True
    
    def prepare_group(self, pipeline, image_set_list, grouping, image_numbers):
        """Set up all of the image providers inside the image_set_list
        """
        image = cellprofiler.cpimage.Image(self.__image, self.__mask)
        provider = cellprofiler.cpimage.VanillaImageProvider(self.__image_name,
                                                             image)
        for image_number in image_numbers:
            image_set_list.get_image_set(image_number-1).providers.append(provider)
        return True 

    def run(self,workspace):
        """Run the module (abstract method)
        
        pipeline     - instance of CellProfiler.Pipeline for this run
        image_set    - the images in the image set being processed
        object_set   - the objects (labeled masks) in this image set
        measurements - the measurements for this run
        """
        pass

    def post_run(self, workspace):
        if self.__release_image:
            del self.__image
            del self.__mask
            
    def get_categories(self,pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        return []
      
    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        return []
    
    def get_measurement_images(self,pipeline,object_name,category,measurement):
        """Return a list of image names used as a basis for a particular measure
        """
        return []
    
    def get_measurement_scales(self,pipeline,object_name,category,measurement,image_name):
        """Return a list of scales (eg for texture) at which a measurement was taken
        """
        return []

class InjectObjects(cellprofiler.cpmodule.CPModule):
    """Inject objects with labels into the pipeline"""
    
    module_name = "InjectObjects"
    variable_revision_number = 1

    def __init__(self,object_name, segmented, unedited_segmented=None, small_removed_segmented=None):
        """Initialize the module with the objects for the object set
        
        object_name - name of the objects to be provided
        segmented   - labels for the segmentation of the image
        unedited_segmented - labels including small and boundary, default =
                             same as segmented
        small_removed_segmented - labels with small objects removed, default = 
                                  same as segmented
        """ 
        super(InjectObjects,self).__init__()
        self.object_name = cellprofiler.settings.ObjectNameProvider("text",object_name)
        self.__segmented = segmented
        self.__unedited_segmented = unedited_segmented
        self.__small_removed_segmented = small_removed_segmented
    
    def settings(self):
        return [self.object_name]
    
    def run(self,workspace):
        my_objects = cellprofiler.objects.Objects()
        my_objects.segmented = self.__segmented
        if self.__unedited_segmented != None:
            my_objects.unedited_segmented = self.__unedited_segmented
        if self.__small_removed_segmented != None:
            my_objects.small_removed_segmented = self.__small_removed_segmented
        workspace.object_set.add_objects(my_objects, self.object_name.value)
