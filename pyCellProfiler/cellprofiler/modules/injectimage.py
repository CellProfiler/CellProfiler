"""
InjectImage.py - for testing, this module injects a single image into the image set

"""
__version__="$Revision$"
import cellprofiler.cpmodule
import cellprofiler.cpimage
import cellprofiler.settings

class InjectImage(cellprofiler.cpmodule.CPModule):
    """This module is intended for testing. It injects an image into the
    image set.
    """
    def __init__(self, image_name, image):
        super(InjectImage,self).__init__()
        self.set_module_name("InjectImage")
        self.__image_name = image_name
        self.__image = image
        self.image_name = cellprofiler.settings.NameProvider("Hardwired image name","imagegroup",image_name) 
    
    def visible_variables(self):
        return [self.image_name]
    
    def upgrade_module_from_revision(self,variable_revision_number):
        """Possibly rewrite the variables in the module to upgrade it to its current revision number
        
        """
        raise NotImplementedError("Please implement UpgradeModuleFromRevision")
    
    def get_help(self):
        """Return help text for the module
        
        """
        raise NotImplementedError("Please implement GetHelp in your derived module class")
            
    variable_revision_number = 1
    
    def annotations(self):
        """Return the variable annotations, as read out of the module file.
        
        Return the variable annotations, as read out of the module file.
        Each annotation is an instance of the Cellprofiler.Settings.Annotation
        class.
        """
        return []
    
    def write_to_handles(self,handles):
        """Write out the module's state to the handles
        
        """
    
    def write_to_text(self,file):
        """Write the module's state, informally, to a text file
        """
    
    def prepare_run(self, pipeline, image_set_list):
        """Set up all of the image providers inside the image_set_list
        """
        image = cellprofiler.cpimage.Image(self.__image)
        image_set_list.get_image_set(0).providers.append(cellprofiler.cpimage.VanillaImageProvider(self.__image_name,image)) 

    def run(self,workspace):
        """Run the module (abstract method)
        
        pipeline     - instance of CellProfiler.Pipeline for this run
        image_set    - the images in the image set being processed
        object_set   - the objects (labeled masks) in this image set
        measurements - the measurements for this run
        """
        pass

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
        
