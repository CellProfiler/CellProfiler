'''imagej2 - in-process connection to ImageJ 2.0

'''

__version__ = "$Revision: 1 $"

import numpy as np
import os
import sys
import time

import cellprofiler.utilities.jutil as J

'''Field type = integer'''
FT_INTEGER = "INTEGER"
'''Field type = floating point'''
FT_FLOAT = "FLOAT"
'''Field type = string'''
FT_STRING = "STRING"
'''Field type = image'''
FT_IMAGE = "IMAGE"
'''Field type = boolean'''
FT_BOOL = "BOOL"
'''Field type = overlay'''
FT_OVERLAY = "OVERLAY"
'''Field type = java.io.File'''
FT_FILE = "FILE"
'''Field type = imagej.util.ColorRGB'''
FT_COLOR = "COLOR"

field_mapping = {
    'java.io.File': FT_FILE,
    'java.lang.Byte': FT_INTEGER,
    'java.lang.Short': FT_INTEGER,
    'java.lang.Integer': FT_INTEGER,
    'java.lang.Long': FT_INTEGER,
    'int': FT_INTEGER,
    'short': FT_INTEGER,
    'byte': FT_INTEGER,
    'long': FT_INTEGER,
    'java.lang.Float': FT_FLOAT,
    'java.lang.Double': FT_FLOAT,
    'float': FT_FLOAT,
    'double': FT_FLOAT,
    'java.lang.String': FT_STRING,
    'java.lang.Boolean': FT_BOOL,
    'boolean': FT_BOOL,
    'imagej.util.ColorRGB': FT_COLOR
}

field_class_mapping = (
    (J.class_for_name('imagej.display.Display'), FT_IMAGE),
    (J.class_for_name('imagej.data.Dataset'), FT_IMAGE),
    (J.class_for_name('imagej.data.roi.Overlay'), FT_OVERLAY) )

def run_imagej(*args):
    J.static_call("imagej/Main", "main", "([Ljava/lang/String;)V",
                  *[unicode(arg) for arg in args])

def create_planar_img(a):
    '''Create a PlanarImg from a numpy array
    
    a - numpy array. The values should be scaled to be between 0 and 255
    
    returns a PlanarImg of double valueswith the same dimensions as the array.
    '''
    a = a.astype(np.float64)
    creator = J.make_instance("net/imglib2/img/basictypeaccess/array/DoubleArray",
                              "(I)V", 0)
    planar_img = J.make_instance(
        "net/imglib2/img/planar/PlanarImg",
        "(Lnet/imglib2/img/basictypeaccess/array/ArrayDataAccess;[JI)V",
        creator, np.array(a.shape), 1)
    def copy_plane(index, src):
        p = J.call(planar_img, "getPlane", "(I)Ljava/lang/Object;", index)
        dest = J.call(p, "getCurrentStorageArray", "()Ljava/lang/Object;")
        length = np.prod(src.shape)
        src = J.get_nice_arg(src, "[D")
        J.static_call("java/lang/System", "arraycopy",
                      "(Ljava/lang/Object;ILjava/lang/Object;II)V",
                      src, 0, dest, 0, length)
    a.shape = (a.shape[0], a.shape[1], np.prod(a.shape[2:]))
    for i in range(a.shape[2]):
        copy_plane(i, a[:,:,i])
    return planar_img

def create_img_plus(a, name):
    '''Create an ImagePlus from a numpy array
    
    a - numpy array. The values should be scaled to the range 0-255
    
    name - a user-visible name for the image
    
    returns an ImagePlus. The metadata axes will be Y, X and channel (if 3-d)
    '''
    x = J.get_static_field("net/imglib2/img/Axes", "X", 
                           "Lnet/imglib2/img/Axes;")
    y = J.get_static_field("net/imglib2/img/Axes", "Y", 
                           "Lnet/imglib2/img/Axes;")
    c = J.get_static_field("net/imglib2/img/Axes", "CHANNEL", 
                           "Lnet/imglib2/img/Axes;")
    img = create_planar_img(a)
    
    img_plus = J.make_instance(
        "net/imglib2/img/ImgPlus",
        "(Lnet/imglib2/img/Img;Ljava/lang/String;[Lnet/imglib2/img/Axis;)V",
        img, name, [y, x] if a.ndim == 2 else [y, x, c])
    
def create_context(service_classes):
    '''Create an ImageJ context for getting services'''
    class Context(object):
        def __init__(self):
            classes = [ J.class_for_name(x) for x in service_classes]
            self.o = J.run_in_main_thread(
                lambda :J.static_call(
                "imagej/ImageJ", "createContext", 
                "([Ljava/lang/Class;)Limagej/ImageJ;", classes), True)
        
        def loadService(self, class_name):
            '''Load the service class with the given class name
            
            You can use this method to pick specific implementations such as
            the headless or Swing UI.
            
            class_name - class name in dotted form, e.g. java.lang.String
            '''
            klass = J.class_for_name(class_name)
            J.call(self.o, 'loadService', '(Ljava/lang/Class;)V', klass)
            
        def getService(self, class_name):
            '''Get a service with the given class name
            
            class_name - class name in dotted form
            
            returns the class or None if no implementor loaded.
            '''
            klass = J.class_for_name(class_name)
            return J.call(self.o, 'getService', 
                          '(Ljava/lang/Class;)Limagej/IService;', klass)
    return Context()

def get_module_service(context):
    '''Get the module service for a given context
    
    context - the instance of ImageJ created by create_context
    
    returns a module service
    '''
    o = context.getService('imagej.ext.module.ModuleService')
    class ModuleItem(object):
        def __init__(self, instance):
            self.o = instance
            
        IV_NORMAL = J.get_static_field("imagej/ext/module/ItemVisibility",
                                       "NORMAL",
                                       "Limagej/ext/module/ItemVisibility;")
        IV_TRANSIENT = J.get_static_field("imagej/ext/module/ItemVisibility",
                                          "TRANSIENT",
                                          "Limagej/ext/module/ItemVisibility;")
        IV_INVISIBLE = J.get_static_field("imagej/ext/module/ItemVisibility",
                                          "INVISIBLE",
                                          "Limagej/ext/module/ItemVisibility;")
        IV_MESSAGE = J.get_static_field("imagej/ext/module/ItemVisibility",
                                        "MESSAGE",
                                        "Limagej/ext/module/ItemVisibility;")
        
        WS_DEFAULT = J.get_static_field("imagej/ext/module/ui/WidgetStyle",
                                        "DEFAULT",
                                        "Limagej/ext/module/ui/WidgetStyle;")
        WS_NUMBER_SPINNER = J.get_static_field(
            "imagej/ext/module/ui/WidgetStyle",
            "NUMBER_SPINNER",
            "Limagej/ext/module/ui/WidgetStyle;")
        WS_NUMBER_SLIDER = J.get_static_field(
            "imagej/ext/module/ui/WidgetStyle",
            "NUMBER_SLIDER",
            "Limagej/ext/module/ui/WidgetStyle;")
        WS_NUMBER_SCROLL_BAR = J.get_static_field(
            "imagej/ext/module/ui/WidgetStyle",
            "NUMBER_SCROLL_BAR",
            "Limagej/ext/module/ui/WidgetStyle;")
        
        
        def getType(self):
            jtype = J.call(self.o, "getType", "()Ljava/lang/Class;")
            type_name = J.call(jtype, "getCanonicalName", "()Ljava/lang/String;")
            if field_mapping.has_key(type_name):
                return field_mapping[type_name]
            for class_instance, result in field_class_mapping:
                if J.call(class_instance, "isAssignableFrom",
                          "(Ljava/lang/Class;)Z", jtype):
                    return result
            return None
            
            
        getWidgetStyle = J.make_method("getWidgetStyle",
                                       "()Limagej/ext/module/ui/WidgetStyle;")
        getMinimumValue = J.make_method("getMinimumValue",
                                        "()Ljava/lang/Object;")
        getMaximumValue = J.make_method("getMaximumValue",
                                        "()Ljava/lang/Object;")
        getStepSize = J.make_method("getStepSize",
                                    "()Ljava/lang/Number;")
        getColumnCount = J.make_method("getColumnCount", "()I")
        getChoices = J.make_method("getChoices", "()Ljava/util/List;")
        getValue = J.make_method("getValue", 
                                 "(Limagej/ext/module/Module;)Ljava/lang/Object;")
        getName = J.make_method("getName", "()Ljava/lang/String;")
        getLabel = J.make_method("getLabel", "()Ljava/lang/String;")
        getDescription = J.make_method("getDescription", "()Ljava/lang/String;")
        
    class ModuleInfo(object):
        def __init__(self, instance):
            self.o = instance

        def getInput(self, name):
            "Gets the input item with the given name."
            return ModuleItem(J.call(
                self.o, "getInput", 
                "(Ljava/lang/String;)Limagej/ext/module/ModuleItem;", name))

        def getOutput(self, name):
            "Gets the output item with the given name."
            return ModuleItem(J.call(
                self.o, "getOutput", 
                "(Ljava/lang/String;)Limagej/ext/module/ModuleItem;", name))

        def getInputs(self):
            inputs = J.call(self.o, "inputs", "()Ljava/lang/Iterable;")
            input_iterator = J.call(inputs, "iterator", "()Ljava/util/Iterator;")
            return [ModuleItem(o) for o in J.iterate_java(input_iterator)]
            
        def getOutputs(self):
            outputs = J.call(self.o, "outputs", "()Ljava/lang/Iterable;")
            output_iterator = J.call(outputs, "iterator", "()Ljava/util/Iterator;")
            return [ModuleItem(o) for o in J.iterate_java(output_iterator)]
        
        getTitle = J.make_method(
            "getTitle",
            "()Ljava/lang/String;")
        createModule = J.make_method(
            "createModule",
            "()Limagej/ext/module/Module;")
        
    class ModuleService(object):
        def __init__(self):
            self.o = o
        def getModules(self):
            modules = J.call(o, "getModules", "()Ljava/util/List;")
            if modules is None:
                return []
            module_iterator = J.call(modules, "iterator", 
                                     "()Ljava/util/Iterator;")
            return [ModuleInfo(x) for x in J.iterate_java(module_iterator)]
        
        def run(self, module,
                pre = None,
                post = None,
                separateThread = False,
                **kwargs):
            '''Run a module
            
            module - the module to run
            
            pre - list of PreprocessorPlugins to run before running module
            
            post - list of PostprocessorPlugins to run after running module
            
            *kwargs - names and values for input parameters
            '''
            input_map = J.get_dictionary_wrapper(
                J.make_instance('java/util/HashMap', '()V'))
            for k,v in kwargs.iteritems():
                input_map.put(k, v)
            if pre is not None:
                pre = J.static_call("java/util/Arrays", "asList",
                                    "([Ljava/lang/Object;)Ljava/util/List;",
                                    pre)
            if post is not None:
                post = J.static_call("java/util/Arrays", "asList",
                                     "([Ljava/lang/Object;)Ljava/util/List;",
                                     post)
            future = J.call(
                self.o, "run", 
                "(Limagej/ext/module/Module;Ljava/util/List;Ljava/util/List;Ljava/util/Map;)Ljava/util/concurrent/Future;",
                module, pre, post, input_map)
            return J.call(
                self.o, "waitFor", 
                "(Ljava/util/concurrent/Future;)Limagej/ext/module/Module;",
                future)
    return ModuleService()

def wrap_module(module):
    class Module(object):
        def __init__(self, o = module):
            self.o = o
            
        getInfo = J.make_method("getInfo", "()Limagej/ext/module/ModuleInfo;")
        getInput = J.make_method("getInput", "(Ljava/lang/String;)Ljava/lang/Object;")
        getOutput = J.make_method("getOutput", "(Ljava/lang/String;)Ljava/lang/Object;")
        setInput = J.make_method("setInput", "(Ljava/lang/String;Ljava/lang/Object;)V")
        setOutput = J.make_method("setOutput", "(Ljava/lang/String;Ljava/lang/Object;)V")
        isResolved = J.make_method("isResolved", "(Ljava/lang/String;)Z")
        setResolved = J.make_method("setResolved", "(Ljava/lang/String;Z)V")
    return Module()

def get_display_service(context):
    '''Get the display service for a given context
    
    context - the ImageJ context for the thread
    '''
    o = context.getService('imagej.display.DisplayService')
    class DisplayService(object):
        def __init__(self):
            self.o = o
        def createDisplay(self, dataset):
            '''Create a display that contains the given dataset'''
            display = J.call(
                self.o,
                "createDisplay", 
                "(Limagej/data/Dataset;)Limagej/display/ImageDisplay;",
                dataset.o)
            return wrap_display(display)
        def getActiveDataset(self, display):
            ds = J.call(self.o, "getActiveDataset",
                        "(Limagej/display/ImageDisplay;)Limagej/data/Dataset;",
                        display.o)
            return wrap_dataset(ds)
        def getActiveDisplay(self):
            return wrap_display(J.call(self.o, "getActiveDisplay",
                                       "()Limagej/display/Display;"))
        def getActiveImageDisplay(self):
            return wrap_display(J.call(self.o, "getActiveImageDisplay",
                                       "()Limagej/display/ImageDisplay;"))
        setActiveDisplay = J.make_method("setActiveDisplay",
                                         "(Limagej/display/Display;)V")
        getGlobalActiveDatasetView = J.make_method(
            "getActiveDatasetView",
            "()Limagej/display/DatasetView;",
            "Get the active display's active dataset view")
        getActiveDatasetView = J.make_method(
            "getActiveDatasetView",
            "(Limagej/display/ImageDisplay;)Limagej/display/DatasetView;")
        getDisplays = J.make_method("getDisplays", "()Ljava/util/List;")
        getImageDisplays = J.make_method("getImageDisplays",
                                         "()Ljava/util/List;")
        getDisplay = J.make_method(
            "getDisplay",
            "(Ljava/lang/String;)Limagej/display/Display;")
        getObjectDisplays = J.make_method(
            "getDisplays", "(Limagej/data/DataObject;)Ljava/util/List;",
            "Get all displays attached to a given data object")
        isUniqueName = J.make_method("isUniqueName", "(Ljava/lang/String;)Z")
        
    return DisplayService()

def wrap_display(display):
    class ImageDisplay(object):
        def __init__(self, o = display):
            self.o = o
        #
        # Display methods
        #
        getDisplayPanel = J.make_method(
            "getDisplayPanel",
            "()Limagej/display/DisplayPanel;")
        #
        # ImageDisplay methods
        #
        canDisplay = J.make_method(
            "canDisplay", "(Limagej/data/Dataset;)Z",
            "Return true if display can display dataset")
        displayDataset = J.make_method(
            "display", "(Limagej/data/Dataset;)V",
            "Display the given dataset (create a view for it)")
        displayOverlay = J.make_method(
            "display", "(Limagej/data/roi/Overlay;)V",
            "Display the given overlay in a view")
        update = J.make_method("update", "()V",
                               "Signal display change")
        addView = J.make_method(
            "addView", "(Limagej/display/DisplayView;)V")
        removeView = J.make_method(
            "removeView", "(Limagej/display/DisplayView;)V")
        removeAllViews = J.make_method("removeAllViews", "()V")
        getViews = J.make_method("getViews", "()Ljava/util/List;")
        getActiveView = J.make_method(
            "getActiveView", "()Limagej/display/DisplayView;")
        getActiveAxis = J.make_method(
            "getActiveAxis", "()Lnet/imglib2/img/Axis;")
        setActiveAxis = J.make_method(
            "setActiveAxis", "(Lnet/imglib2/img/Axis;)V")
        redoWindowLayout = J.make_method("redoWindowLayout", "()V")
        getImageCanvas = J.make_method(
            "getImageCanvas", "()Limagej/display/ImageCanvas;")
        getAxes = J.make_method("getAxes", "()Ljava/util/List;")
    return ImageDisplay()
                
def wrap_display_panel(display_panel):
    class DisplayPanel(object):
        def __init__(self, o = display_panel):
            self.o = o
        getDisplay = J.make_method("getDisplay",
                                   "()Limagej/display/Display;")
        addEventDispatcher = J.make_method(
            "addEventDispatcher", "(Limagej/display/EventDispatcher;)V")
        close = J.make_method("close", "()V")
        makeActive = J.make_method("makeActive", "()V")
        redoLayout = J.make_method("redoLayout", "()V")
        setLabel = J.make_method("setLabel", "(Ljava/lang/String;)V")
        setTitle = J.make_method("setTitle", "(Ljava/lang/String;)V")
        update = J.make_method("update", "()V")
    return DisplayPanel()

def get_overlay_service(context):
    '''Get the context's overlay service'''
    o = context.getService('imagej.display.OverlayService')
    class OverlayService(object):
        def __init__(self, o=o):
            self.o = o
            
        getOverlays = J.make_method("getOverlays", "()Ljava/util/List;")
        getDisplayOverlays = J.make_method(
            "getOverlays",
            "(Limagej/display/ImageDisplay;)Ljava/util/List;")
        addOverlays = J.make_method(
            "addOverlays", 
            "(Limagej/display/ImageDisplay;Ljava/util/List;)V")
        removeOverlay = J.make_method(
            "removeOverlay",
            "(Limagej/display/ImageDisplay;Limagej/data/roi/Overlay;)V")
        getSelectionBounds = J.make_method(
            "getSelectionBounds",
            "(Limagej/display/ImageDisplay;)Limagej/util/RealRect;")
    return OverlayService()

class Axes(object):
    '''Represents the net.imglib2.img.Axes enum'''
    
    def get_named_axis(self, axis_name):
        return J.get_static_field("net/imglib2/img/Axes", axis_name, 
                                  "Lnet/imglib2/img/Axes;")
    @property
    def X(self):
        return self.get_named_axis("X")
    
    @property
    def Y(self):
        return self.get_named_axis("Y")
        
    @property
    def CHANNEL(self):
        return self.get_named_axis("CHANNEL")
    
def create_dataset(pixel_data, name = None, axes = None):
    '''Create a dataset from a numpy array
    
    pixel_data - numpy array where index 0 is the I or Y axis, index 1 is the
                 J or X axis and index 2, if it exists, is the channel axis.
                 
    name - optional name for the dataset
    '''
    if axes is None:
        if pixel_data.ndim == 2:
            axes = [Axes().X, Axes().Y]
            pixel_data = pixel_data.transpose((1,0))
        else:
            axes = [Axes().X, Axes().Y, Axes().CHANNEL]
            pixel_data = pixel_data.transpose((1,0,2))
    #
    # Create a dataset of the correct shape, with the correct axes.
    # We make a 64-bit floating point image.
    #
    dataset = J.static_call(
        "imagej/data/Dataset",
        "create",
        "([JLjava/lang/String;[Lnet/imglib2/img/Axis;IZZ)Limagej/data/Dataset;",
        np.array(pixel_data.shape), name, axes, 64, True, True)
    dataset = wrap_dataset(dataset)
    imgplus = dataset.getImgPlus()
    #
    # Now use a copying utility to fill the imgplus with array data
    #
    strides = np.cumprod([1] + list(pixel_data.shape[:-1]))
    J.static_call("net/imglib2/util/ImgUtil", "copy",
                  "([DI[ILnet/imglib2/img/Img;)V",
                  pixel_data.flatten(), 0, strides, imgplus)
    return dataset

def create_overlay(mask):
    '''Create a bitmask overlay from a numpy boolean array
    
    mask - boolean numpy array organized as i,j = y,x
    '''
    assert mask.ndim == 2
    mask = mask.transpose()
    strides = np.array([1, mask.shape[0]])
    
    imgFactory = J.make_instance(
        "net/imglib2/img/planar/PlanarImgFactory", "()V")
    bit_type = J.make_instance("net/imglib2/type/logic/BitType", "()V")
    img = J.call(
        imgFactory, "create", 
        "([JLnet/imglib2/type/NativeType;)Lnet/imglib2/img/planar/PlanarImg;",
        np.array(mask.shape), bit_type)
    
    J.static_call("net/imglib2/util/ImgUtil", 
                  "copy", "([ZI[ILnet/imglib2/img/Img;)V",
                  mask.flatten(), 0, strides, img)
    roi = J.make_instance(
        "net/imglib2/roi/BinaryMaskRegionOfInterest",
        "(Lnet/imglib2/img/Img;)V", img)
    overlay = J.make_instance(
        "imagej/data/roi/BinaryMaskOverlay",
        "(Lnet/imglib2/roi/BinaryMaskRegionOfInterest;)V", roi)
    return overlay

def wrap_dataset(dataset):
    
    class Dataset(object):
        def __init__(self, o=dataset):
            self.o = o
        getImgPlus = J.make_method("getImgPlus", "()Lnet/imglib2/img/ImgPlus;")
        setImgPlus = J.make_method("setImgPlus","(Lnet/imglib2/img/ImgPlus;)V")
        getAxes = J.make_method("getAxes","()[Lnet/imglib2/img/Axis;")
        getType = J.make_method("getType", "()Lnet/imglib2/type/numeric/RealType;")
        isSigned = J.make_method("isSigned", "()Z")
        isInteger = J.make_method("isInteger", "()Z")
        getName = J.make_method("getName", "()Ljava/lang/String;")
        setName = J.make_method("setName","(Ljava/lang/String;)V")
        calibration = J.make_method("calibration", "(I)D")
        setCalibration = J.make_method("setCalibration", "(DI)V")
        def get_pixel_data(self, axes = None):
            imgplus = self.getImgPlus()
            pixel_data = get_pixel_data(imgplus)
            inv_axes = J.get_env().get_object_array_elements(self.getAxes())
            if axes is None:
                axes = [ Axes().Y, Axes().X]
                if len(inv_axes) > 2:
                    axes.append(Axes().CHANNEL)
            transpose = []
            for axis in axes:
                matches = [i for i, inv_axis in enumerate(inv_axes)
                           if J.call(inv_axis, "equals", 
                                     "(Ljava/lang/Object;)Z", axis)]
                if len(matches) != 1:
                    raise ValueError("No match for %s axis" % J.to_string(axis))
                transpose.append(matches[0])
            return pixel_data.transpose(transpose)
    return Dataset()

def get_pixel_data(img):
    '''Get the pixel data from an image'''
    interval = wrap_interval(img)
    dims = interval.dimensions()
    #
    # Make a Java double array
    #
    a = np.zeros(np.prod(dims), np.float64)
    ja = J.get_env().make_double_array(np.ascontiguousarray(a))
    strides = np.cumprod([1] + list(dims[:-1]))
    J.static_call("net/imglib2/util/ImgUtil", "copy", 
                  "(Lnet/imglib2/img/Img;[DI[I)V",
                  img, ja, 0, strides)
    a = J.get_env().get_double_array_elements(ja)
    a.shape = dims
    return a
        
def wrap_interval(interval):
    '''Return a class wrapper around a net.imglib2.Interval'''
    class Interval(object):
        def __init__(self, o = interval):
            self.o = o
            
        numDimensions = J.make_method("numDimensions", "()I")
        min1D = J.make_method("min", "(I)J", 
                              "Retrieve the minimum coordinate for a single dimension")
        max1D = J.make_method("max", "(I)J",
                              "Retrieve the maximum coordinate for a single dimension")
        dimension = J.make_method("dimension", "(I)J",
                                  "Retrieve the number of pixels in the given dimension")
        def minND(self):
            return [self.min1D(i) for i in range(self.numDimensions())]
        
        def maxND(self):
            return [self.max1D(i) for i in range(self.numDimensions())]
        
        def dimensions(self):
            return [self.dimension(i) for i in range(self.numDimensions())]
    return Interval()


def make_color_rgb_from_html(s):
    '''Make an imagej.util.ColorRGB from an HTML color
    
    HTML colors have the form, #rrggbb or are one of the names
    from the CSS-3 colors.
    '''
    return J.static_call("fromHTMLColor", 
                         "(Ljava/lang/String;)Limagej/util/ColorRGB;", s)
    

def color_rgb_to_html(color_rgb):
    '''Return an HTML-encoded color value from an imagej.util.ColorRGB
    
    color_rgb - a Java imagej.util.ColorRGB object
    '''
    return J.call(color_rgb, "toHTMLColor()", "()Ljava/lang/String;")
    
if __name__=="__main__":
    classpath = os.path.join(os.path.split(__file__)[0], "imagej-2.0-SNAPSHOT-all.jar")
    J.start_vm(["-Djava.class.path="+classpath])
    my_context = create_context([
            "imagej.event.EventService",
            "imagej.object.ObjectService",
            "imagej.platform.PlatformService",
            "imagej.ext.plugin.PluginService",
            "imagej.ext.module.ModuleService"
        ])
    module_service = get_module_service(my_context)
    module_infos = module_service.getModules()
    for module_info in module_infos:
        print J.to_string(module_info.o)
