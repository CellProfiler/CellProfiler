"""colortogray.py 

This module converts an image with 3 color channels to an NxM
grayscale image
"""
__version__="$Revision$"

import wx
import matplotlib.cm
import matplotlib.backends.backend_wxagg


import cellprofiler.cpmodule as cpm
import cellprofiler.variable as cpv
import cellprofiler.cpimage  as cpi

COMBINE = "Combine"
SPLIT = "Split"

class ColorToGray(cpm.AbstractModule):
    """Module to convert a 3-color image to grayscale"""
    
    def __init__(self):
        super(ColorToGray,self).__init__()
        
        self.module_name = "ColorToGray"
        self.image_name = cpv.NameSubscriber("What did you call the image to be converted to Gray?",
                                             "imagegroup","None")
        self.combine_or_split = cpv.Choice("How do you want to convert the color image?",
                                           [COMBINE,SPLIT])
        self.grayscale_name = cpv.NameProvider("What do you want to call the resulting grayscale image?",
                                               "imagegroup","OrigGray")
        self.red_contribution = cpv.Float("Enter the relative contribution of the red channel",
                                          1,0)
        self.green_contribution = cpv.Float("Enter the relative contribution of the green channel",
                                            1,0)
        self.blue_contribution = cpv.Float("Enter the relative contribution of the blue channel",
                                           1,0)
        self.use_red = cpv.Binary('Create an image from the red channel?',True)
        self.red_name = cpv.NameProvider('What do you want to call the image that was red?',
                                         "imagegroup","OrigRed")
        self.use_green = cpv.Binary('Create an image from the green channel?',True)
        self.green_name = cpv.NameProvider('What do you want to call the image that was green?',
                                         "imagegroup","OrigGreen")
        self.use_blue = cpv.Binary('Create an image from the blue channel?',True)
        self.blue_name = cpv.NameProvider('What do you want to call the image that was blue?',
                                         "imagegroup","OrigBlue")

    variable_revision_number = 1
    
    def variables(self):
        """Return all of the variables in the serialization order"""
        return [self.image_name, self.combine_or_split,
                self.grayscale_name, self.red_contribution,
                self.green_contribution, self.blue_contribution,
                self.use_red, self.red_name,
                self.use_green, self.green_name,
                self.use_blue, self.blue_name
                ]
    
    def visible_variables(self):
        """Return either the "combine" or the "split" variables"""
        vv = [self.image_name, self.combine_or_split]
        if self.should_combine:
            vv.extend([self.grayscale_name, self.red_contribution,
                       self.green_contribution, self.blue_contribution])
        else:
            for v_use,v_name in ((self.use_red  ,self.red_name),
                                 (self.use_green,self.green_name),
                                 (self.use_blue ,self.blue_name)):
                vv.append(v_use)
                if v_use.value:
                    vv.append(v_name)
        return vv
    
    def set_variable_values(self,variable_values,variable_revision_number,module_name):
        if module_name == "ColorToGray" and variable_revision_number == 1:
            new_variable_values = [ variable_values[0],  # image name
                                    variable_values[1],  # combine or split
                                                         # blank slot for text: "Combine options"
                                    variable_values[3],  # grayscale name
                                    variable_values[4],  # red contribution
                                    variable_values[5],  # green contribution
                                    variable_values[6]   # blue contribution
                                                         # blank slot for text: "Split options"
                                    ]
            for i in range(3):
                vv = variable_values[i+8]
                use_it = ((vv == cpv.DO_NOT_USE or vv == "N") and cpv.NO) or cpv.YES
                new_variable_values.append(use_it)
                new_variable_values.append(vv)
            variable_values = new_variable_values
            module_name = self.module_class()
            variable_revision_number = 1
            
        super(ColorToGray,self).set_variable_values(variable_values, variable_revision_number, module_name)
        
    def get_should_combine(self):
        """True if we are supposed to combine RGB to gray"""
        return self.combine_or_split == COMBINE
    should_combine = property(get_should_combine)
    
    def get_should_split(self):
        """True if we are supposed to split each color into an image"""
        return self.combine_or_split == SPLIT
    should_split = property(get_should_split)
    
    def category(self):
        return "Image Processing"
    
    def test_valid(self,pipeline):
        """Test to see if the module is in a valid state to run
        
        Throw a ValidationError exception with an explanation if a module is not valid.
        Make sure that we output at least one image if split
        """
        super(ColorToGray,self).test_valid(pipeline)
        if self.should_split and not any([self.use_red.value, self.use_blue.value, self.use_green.value]):
            raise cpv.ValidationError("You must output at least one of the color images when in split mode",self.use_red)
    
    def run(self,pipeline,image_set,object_set,measurements, frame):
        """Run the module
        
        pipeline     - instance of CellProfiler.Pipeline for this run
        image_set    - the images in the image set being processed
        object_set   - the objects (labeled masks) in this image set
        measurements - the measurements for this run
        frame        - display within this frame (or None to not display)
        """
        image = image_set.get_image(self.image_name)
        if len(image.image.shape) == 2:
            raise ValueError("The image is already grayscale")
        
        if self.should_combine:
            self.run_combine(pipeline,image_set, object_set, measurements, frame, image)
        else:
            self.run_split(pipeline,image_set, object_set, measurements, frame, image)
    
    def run_combine(self, pipeline, image_set, object_set, measurements, frame, image):
        """Combine images to make a grayscale one
        """
        input_image  = image.image
        denominator = self.red_contribution.value + self.green_contribution.value + self.blue_contribution.value 

        output_image = (input_image[:,:,0] * self.red_contribution.value + \
                        input_image[:,:,1] * self.green_contribution.value + \
                        input_image[:,:,2] * self.blue_contribution.value) / denominator
        if image.has_mask:
            image = cpi.Image(output_image,image.mask)
        else:
            image = cpi.Image(output_image)
        image_provider = cpi.VanillaImageProvider(self.grayscale_name,image)
        image_set.providers.append(image_provider)
        
        if frame:
            self.display_combine(frame, input_image, output_image)
        
    
    def display_combine(self, frame, input_image, output_image):
        window_name = "CellProfiler(%s:%d)Combine"%(self.module_name,self.module_num)
        my_frame=frame.FindWindowByName(window_name)
        if not my_frame:
            class my_frame_class(wx.Frame):
                def __init__(self):
                    wx.Frame.__init__(self,frame,-1,"Color to Gray",name=window_name)
                    sizer = wx.BoxSizer()
                    self.figure = figure= matplotlib.figure.Figure()
                    self.panel  = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(self,-1,self.figure) 
                    self.SetSizer(sizer)
                    sizer.Add(self.panel,1,wx.EXPAND)
                    self.Bind(wx.EVT_PAINT,self.on_paint)
                    self.input_axes = self.figure.add_subplot(1,2,1)
                    self.output_axes = self.figure.add_subplot(1,2,2)
                    self.Fit()
                    self.Show()
                def on_paint(self, event):
                    dc = wx.PaintDC(self)
                    self.panel.draw(dc)

            my_frame = my_frame_class()
            
        my_frame.input_axes.clear()
        my_frame.input_axes.imshow(input_image)
        my_frame.input_axes.set_title("Original image")
        
        my_frame.output_axes.clear()
        my_frame.output_axes.imshow(output_image,matplotlib.cm.Greys_r)
        my_frame.output_axes.set_title("Grayscale image")
        my_frame.Refresh()
        
    def run_split(self, pipeline, image_set, object_set, measurements, frame, image):
        """Combine images to make a grayscale one
        """
        input_image  = image.image
        disp_collection = []
        for index, v_use, v_name, title in ((0, self.use_red, self.red_name, "Red"),
                                            (1, self.use_green, self.green_name, "Green"),
                                            (2, self.use_blue, self.blue_name, "Blue")):
            if v_use.value:
                output_image = input_image[:,:,index]
                if image.has_mask:
                    image = cpi.Image(output_image,image.mask)
                else:
                    image = cpi.Image(output_image)
                image_provider = cpi.VanillaImageProvider(v_name.value,image)
                image_set.providers.append(image_provider)
                disp_collection.append([output_image, title])
        if frame:
            self.display_split(frame, input_image, disp_collection)
                
    
    def display_split(self, frame, input_image, disp_collection):
        window_name = "CellProfiler(%s:%d)Split%d"%(self.module_name,self.module_num,len(disp_collection))
        my_frame=frame.FindWindowByName(window_name)
        if not my_frame:
            class my_frame_class(wx.Frame):
                def __init__(self):
                    wx.Frame.__init__(self,frame,-1,"Color to Gray",name=window_name)
                    sizer = wx.BoxSizer()
                    self.figure = figure= matplotlib.figure.Figure()
                    self.panel  = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(self,-1,self.figure) 
                    self.SetSizer(sizer)
                    sizer.Add(self.panel,1,wx.EXPAND)
                    self.Bind(wx.EVT_PAINT,self.on_paint)
                    if len(disp_collection) == 1:
                        self.input_axes = self.figure.add_subplot(1,2,1)
                        self.output_axes = [self.figure.add_subplot(1,2,2)]
                    else:
                        self.input_axes = self.figure.add_subplot(2,2,1)
                        self.output_axes = [self.figure.add_subplot(2,2,i+2) for i in range(len(disp_collection))]
                    self.Fit()
                    self.Show()
                def on_paint(self, event):
                    dc = wx.PaintDC(self)
                    self.panel.draw(dc)

            my_frame = my_frame_class()
            
        my_frame.input_axes.clear()
        my_frame.input_axes.imshow(input_image)
        my_frame.input_axes.set_title("Original image")
        
        for i, disp in zip(range(len(disp_collection)),disp_collection):
            my_frame.output_axes[i].clear()
            my_frame.output_axes[i].imshow(disp[0],matplotlib.cm.Greys_r)
            my_frame.output_axes[i].set_title("%s image"%(disp[1]))
        my_frame.Refresh()
    
    
        
    def get_help(self):
        return """SHORT DESCRIPTION:
Converts RGB (Red, Green, Blue) color images to grayscale. All channels
can be merged into one grayscale image (COMBINE option) or each channel 
can be extracted into a separate grayscale image (SPLIT option).
*************************************************************************
Note: this module is especially helpful because all identify modules
require grayscale images.

Settings:

Split:
Takes a color image and splits the three channels (red, green, blue) into
three separate grayscale images.

Combine:
Takes a color image and converts it to grayscale by combining the three
channels (red, green, blue) together.

Adjustment factors: Leaving the adjustment factors set to 1 will balance
all three colors equally in the final image, which will use the same
range of intensities as the incoming image.  To weight colors relative to
each other, the adjustment factor can be increased (to increase the
weighting) or decreased (to decrease the weighting).

See also GrayToColor.
"""
