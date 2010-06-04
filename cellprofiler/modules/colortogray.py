'''
<b> Color to Gray</b> converts an image with three color channels to one <i>or</i> three grayscale images
<hr>

This module converts RGB (Red, Green, Blue) color images to grayscale. All channels
can be merged into one grayscale image (<i>Combine</i>), or each channel 
can be extracted into a separate grayscale image (<i>Split</i>). If you use <i>Combine</i>, 
the relative weights will adjust the contribution of the colors relative to each other.<br>
<br>
<i>Note:</i>All <b>Identify</b> modules require grayscale images.
<p>See also <b>GrayToColor</b>.
'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org
__version__="$Revision$"


import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage  as cpi

COMBINE = "Combine"
SPLIT = "Split"

class ColorToGray(cpm.CPModule):

    module_name = "ColorToGray"
    variable_revision_number = 1
    category = "Image Processing"
    
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber("Select the input image",
                                             "None")
        self.combine_or_split = cps.Choice("Conversion method",
                                           [COMBINE,SPLIT],doc='''
                                           How do you want to convert the color image? 
                                           <ul>
                                           <li><i>Split:</i> Splits the three channels
                                           (red, green, blue) of a color image into three separate grayscale images. </li>
                                           <li><i>Combine:</i> Converts a color image to a grayscale 
                                           image by combining the three channels (red, green, blue) together.</li>
                                           </ul>''')
        
        # The following settings are used for the combine option
        self.grayscale_name = cps.ImageNameProvider("Name the output image",
                                               "OrigGray")
        self.red_contribution = cps.Float("Relative weight of the red channel",
                                          1,0,doc='''<i>(Used only when combining channels)</i><br>
                                          Relative weights: If all relative weights are equal, all three 
                                          colors contribute equally in the final image. To weight colors relative 
                                          to each other, increase or decrease the relative weights.''')
        
        self.green_contribution = cps.Float("Relative weight of the green channel",
                                            1,0,doc='''<i>(Used only when combining channels)</i><br>
                                            Relative weights: If all relative weights are equal, all three 
                                            colors contribute equally in the final image. To weight colors relative 
                                            to each other, increase or decrease the relative weights.''')
        
        self.blue_contribution = cps.Float("Relative weight of the blue channel",
                                           1,0,doc='''<i>(Used only when combining channels)</i><br>
                                           Relative weights: If all relative weights are equal, all three 
                                           colors contribute equally in the final image. To weight colors relative 
                                           to each other, increase or decrease the relative weights.''')
        
        # The following settings are used for the split option
        self.use_red = cps.Binary('Convert red to gray?',True)
        self.red_name = cps.ImageNameProvider('Name the output image',
                                         "OrigRed")
        
        self.use_green = cps.Binary('Convert green to gray?',True)
        self.green_name = cps.ImageNameProvider('Name the output image',
                                         "OrigGreen")
        
        self.use_blue = cps.Binary('Convert blue to gray?',True)
        self.blue_name = cps.ImageNameProvider('Name the output image',
                                         "OrigBlue")

    def visible_settings(self):
        """Return either the "combine" or the "split" settings"""
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
    
    def settings(self):
        """Return all of the settings in a consistent order"""
        return [self.image_name, self.combine_or_split,
                self.grayscale_name, self.red_contribution,
                self.green_contribution, self.blue_contribution,
                self.use_red, self.red_name,
                self.use_green, self.green_name,
                self.use_blue, self.blue_name
                ]
    
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
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_color=True)
        if self.should_combine():
            self.run_combine(workspace, image)
        else:
            self.run_split(workspace, image)
    
    def display(self, workspace):
        if self.should_combine():
            self.display_combine(workspace)
        else:
            self.display_split(workspace)

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
        image = cpi.Image(output_image,parent_image=image)
        workspace.image_set.add(self.grayscale_name.value,image)

        workspace.display_data.input_image = input_image
        workspace.display_data.output_image = output_image
       
    
    def display_combine(self, workspace):
        import matplotlib.cm

        input_image = workspace.display_data.input_image
        output_image = workspace.display_data.output_image
        figure = workspace.create_or_find_figure(title="Color to gray",
                                                 subplots=(1,2))
        figure.subplot_imshow(0, 0, input_image, 
                              title = "Original image: %s"%(self.image_name))
        figure.subplot_imshow(0, 1, output_image,
                              title = "Grayscale image: %s"%(self.grayscale_name),
                              colormap = matplotlib.cm.Greys_r, 
                              sharex=figure.subplot(0,0),
                              sharey=figure.subplot(0,0))
        
    def run_split(self, workspace, image):
        """Split image into individual components
        """
        input_image  = image.pixel_data
        disp_collection = []
        for index, v_use, v_name, title in ((0, self.use_red, self.red_name, "Red"),
                                            (1, self.use_green, self.green_name, "Green"),
                                            (2, self.use_blue, self.blue_name, "Blue")):
            if v_use.value:
                output_image = input_image[:,:,index]
                image = cpi.Image(output_image,parent_image=image)
                workspace.image_set.add(v_name.value,image)
                disp_collection.append([output_image, title])

        workspace.display_data.input_image = input_image
        workspace.display_data.disp_collection = disp_collection
    
    def display_split(self, workspace):
        import matplotlib.cm

        input_image = workspace.display_data.input_image
        disp_collection = workspace.display_data.disp_collection
        ndisp = len(disp_collection)
        if ndisp == 1:
            subplots = (1,2)
        else:
            subplots = (2,2)
        figure=workspace.create_or_find_figure(title="Color to gray",
                                               subplots=subplots)
        figure.subplot_imshow(0, 0, input_image,
                              title = "Original image")
        
        if ndisp == 1:
            layout = [(0,1)]
        elif ndisp == 2:
            layout = [ (1,0),(0,1)]
        else:
            layout = [(1,0),(0,1),(1,1)]
        for xy, disp in zip(layout, disp_collection):
            figure.subplot_imshow(xy[0], xy[1], disp[0],
                                  title = "%s image"%(disp[1]),
                                  colormap = matplotlib.cm.Greys_r, 
                                  sharex=figure.subplot(0,0),
                                  sharey=figure.subplot(0,0))
        
    def is_interactive(self):
        return False

    def upgrade_settings(self,
                         setting_values,
                         variable_revision_number,
                         module_name,
                         from_matlab):
        if from_matlab and variable_revision_number == 1:
            new_setting_values = [ setting_values[0],  # image name
                                    setting_values[1],  # combine or split
                                                         # blank slot for text: "Combine options"
                                    setting_values[3],  # grayscale name
                                    setting_values[4],  # red contribution
                                    setting_values[5],  # green contribution
                                    setting_values[6]   # blue contribution
                                                         # blank slot for text: "Split options"
                                    ]
            for i in range(3):
                vv = setting_values[i+8]
                use_it = ((vv == cps.DO_NOT_USE or vv == "N") and cps.NO) or cps.YES
                new_setting_values.append(use_it)
                new_setting_values.append(vv)
            setting_values = new_setting_values
            module_name = self.module_class()
            variable_revision_number = 1
            from_matlab = False
            
        return setting_values, variable_revision_number, from_matlab
        
