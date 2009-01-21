"""colortogray.py 

This module converts an image with 3 color channels to an NxM
grayscale image
"""
__version__="$Revision$"

import wx
import matplotlib.cm
import matplotlib.backends.backend_wxagg


import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage  as cpi

import cellprofiler.gui.cpfigure as cpf

COMBINE = "Combine"
SPLIT = "Split"

class ColorToGray(cpm.CPModule):
    """SHORT DESCRIPTION:
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

    variable_revision_number = 1
    category = "Image Processing"
    
    def create_variables(self):
        self.module_name = "ColorToGray"
        self.image_name = cps.NameSubscriber("Which image do you want to convert to gray?",
                                             "imagegroup","None")
        self.combine_or_split = cps.Choice("How do you want to convert the color image?",
                                           [COMBINE,SPLIT])
        
        # The following variables are used for the combine option
        self.grayscale_name = cps.NameProvider("What do you want to call the resulting grayscale image?",
                                               "imagegroup","OrigGray")
        self.red_contribution = cps.Float("Enter the relative contribution of the red channel",
                                          1,0)
        self.green_contribution = cps.Float("Enter the relative contribution of the green channel",
                                            1,0)
        self.blue_contribution = cps.Float("Enter the relative contribution of the blue channel",
                                           1,0)
        
        # The following variables are used for the split option
        self.use_red = cps.Binary('Create an image from the red channel?',True)
        self.red_name = cps.NameProvider('What do you want to call the image that was red?',
                                         "imagegroup","OrigRed")
        self.use_green = cps.Binary('Create an image from the green channel?',True)
        self.green_name = cps.NameProvider('What do you want to call the image that was green?',
                                         "imagegroup","OrigGreen")
        self.use_blue = cps.Binary('Create an image from the blue channel?',True)
        self.blue_name = cps.NameProvider('What do you want to call the image that was blue?',
                                         "imagegroup","OrigBlue")

    def visible_variables(self):
        """Return either the "combine" or the "split" variables"""
        vv = [self.image_name, self.combine_or_split]
        if self.should_combine():
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
    
    def variables(self):
        """Return all of the variables in a consistent order"""
        return [self.image_name, self.combine_or_split,
                self.grayscale_name, self.red_contribution,
                self.green_contribution, self.blue_contribution,
                self.use_red, self.red_name,
                self.use_green, self.green_name,
                self.use_blue, self.blue_name
                ]
    
    def backwards_compatibilize(self,
                                variable_values,
                                variable_revision_number,
                                module_name,
                                from_matlab):
        if from_matlab and variable_revision_number == 1:
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
                use_it = ((vv == cps.DO_NOT_USE or vv == "N") and cps.NO) or cps.YES
                new_variable_values.append(use_it)
                new_variable_values.append(vv)
            variable_values = new_variable_values
            module_name = self.module_class()
            variable_revision_number = 1
            from_matlab = False
            
        return variable_values
        
    def should_combine(self):
        """True if we are supposed to combine RGB to gray"""
        return self.combine_or_split == COMBINE
    
    def should_split(self):
        """True if we are supposed to split each color into an image"""
        return self.combine_or_split == SPLIT
    
    def validate_module(self,pipeline):
        """Test to see if the module is in a valid state to run
        
        Throw a ValidationError exception with an explanation if a module is not valid.
        Make sure that we output at least one image if split
        """
        if self.should_split() and not any([self.use_red.value, self.use_blue.value, self.use_green.value]):
            raise cps.ValidationError("You must output at least one of the color images when in split mode",
                                      self.use_red)
    
    def run(self,workspace):
        """Run the module
        
        pipeline     - instance of CellProfiler.Pipeline for this run
        workspace    - the workspace contains:
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - display within this frame (or None to not display)
        """
        image = workspace.image_set.get_image(self.image_name,
                                              must_be_color=True)
        if self.should_combine():
            self.run_combine(workspace, image)
        else:
            self.run_split(workspace, image)
    
    def run_combine(self, workspace, image):
        """Combine images to make a grayscale one
        """
        input_image  = image.pixel_data
        denominator = (self.red_contribution.value +
                       self.green_contribution.value +
                       self.blue_contribution.value )

        output_image = (input_image[:,:,0] * self.red_contribution.value + 
                        input_image[:,:,1] * self.green_contribution.value + 
                        input_image[:,:,2] * self.blue_contribution.value) / denominator
        image = cpi.Image(output_image,use_mask_from=image)
        workspace.image_set.add(self.grayscale_name,image)
        
        if workspace.display:
            self.display_combine(workspace, input_image, output_image)
        
    
    def display_combine(self, workspace, input_image, output_image):
        figure=workspace.create_or_find_figure(title="Color to gray",
                                               subplots=(1,2))
        input_axes = figure.subplot(0,0)
        input_axes.clear()
        input_axes.imshow(input_image)
        input_axes.set_title("Original image: %s"%(self.image_name))
        
        output_axes = figure.subplot(0,1)
        output_axes.clear()
        output_axes.imshow(output_image,matplotlib.cm.Greys_r)
        output_axes.set_title("Grayscale image: %s"%(self.grayscale_name))
        
    def run_split(self, workspace, image):
        """Combine images to make a grayscale one
        """
        input_image  = image.pixel_data
        disp_collection = []
        for index, v_use, v_name, title in ((0, self.use_red, self.red_name, "Red"),
                                            (1, self.use_green, self.green_name, "Green"),
                                            (2, self.use_blue, self.blue_name, "Blue")):
            if v_use.value:
                output_image = input_image[:,:,index]
                image = cpi.Image(output_image,use_mask_from=image)
                workspace.image_set.add(v_name.value,image)
                disp_collection.append([output_image, title])
        if workspace.display:
            self.display_split(workspace, input_image, disp_collection)
                
    
    def display_split(self, workspace, input_image, disp_collection):
        ndisp = len(disp_collection)
        if ndisp == 1:
            subplots = (1,2)
        else:
            subplots = (2,2)
        figure=workspace.create_or_find_figure(title="Color to gray",
                                               subplots=subplots)

        input_axes = figure.subplot(0,0)
        input_axes.clear()
        input_axes.imshow(input_image)
        input_axes.set_title("Original image")

        if ndisp == 1:
            layout = [(0,1)]
        elif ndisp == 2:
            layout = [ (1,0),(0,1)]
        else:
            layout = [(1,0),(0,1),(1,1)]
        for xy, disp in zip(layout,disp_collection):
            output_axes = figure.subplot(xy[0],xy[1])
            output_axes.clear()
            output_axes.imshow(disp[0],matplotlib.cm.Greys_r)
            output_axes.set_title("%s image"%(disp[1]))
        
