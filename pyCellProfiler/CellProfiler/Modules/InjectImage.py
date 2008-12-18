"""
InjectImage.py - for testing, this module injects a single image into the image set

"""
import CellProfiler.Module
import CellProfiler.Image

class InjectImage(CellProfiler.Module.AbstractModule):
    """Cut and paste this in order to get started writing a module
    """
    def __init__(self, image_name, image):
        CellProfiler.Module.AbstractModule.__init__(self)
        self.SetModuleName("InjectImage")
        self.__image_name = image_name
        self.__image = image
    
    def UpgradeModuleFromRevision(self,variable_revision_number):
        """Possibly rewrite the variables in the module to upgrade it to its current revision number
        
        """
        raise NotImplementedError("Please implement UpgradeModuleFromRevision")
    
    def GetHelp(self):
        """Return help text for the module
        
        """
        raise NotImplementedError("Please implement GetHelp in your derived module class")
            
    def VariableRevisionNumber(self):
        """The version number, as parsed out of the .m file, saved in the handles or rewritten using an import rule
        """
        return 1
    
    def Annotations(self):
        """Return the variable annotations, as read out of the module file.
        
        Return the variable annotations, as read out of the module file.
        Each annotation is an instance of the CellProfiler.Variable.Annotation
        class.
        """
        return []
    
    def WriteToHandles(self,handles):
        """Write out the module's state to the handles
        
        """
    
    def WriteToText(self,file):
        """Write the module's state, informally, to a text file
        """
    
    def PrepareRun(self, pipeline, image_set_list):
        """Set up all of the image providers inside the image_set_list
        """
        image = CellProfiler.Image.Image(self.__image)
        image_set_list.GetImageSet(0).Providers.append(CellProfiler.Image.VanillaImageProvider(self.__image_name,image)) 

    def Run(self,pipeline,image_set,object_set,measurements, frame):
        """Run the module (abstract method)
        
        pipeline     - instance of CellProfiler.Pipeline for this run
        image_set    - the images in the image set being processed
        object_set   - the objects (labeled masks) in this image set
        measurements - the measurements for this run
        """
        pass

    def GetCategories(self,pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        return []
      
    def GetMeasurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        return []
    
    def GetMeasurementImages(self,pipeline,object_name,category,measurement):
        """Return a list of image names used as a basis for a particular measure
        """
        return []
    
    def GetMeasurementScales(self,pipeline,object_name,category,measurement,image_name):
        """Return a list of scales (eg for texture) at which a measurement was taken
        """
        return []
        
