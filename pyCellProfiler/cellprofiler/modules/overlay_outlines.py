'''overlay_outlines.py - module to create images with outlines

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import numpy as np
import uuid

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps

WANTS_COLOR = "Color"
WANTS_GRAYSCALE = "Grayscale"

MAX_IMAGE = "Max of image"
MAX_POSSIBLE = "Max possible"

COLORS = { "White":  (255,255,255),
           "Black":  (0,0,0),
           "Red":    (255,0,0),
           "Green":  (0,255,0),
           "Blue":   (0,0,255),
           "Yellow": (255,255,0) }

COLOR_ORDER = ["Red", "Green", "Blue","Yellow","White","Black"]

class OverlayOutlines(cpm.CPModule):
    ''' SHORT DESCRIPTION:
Places outlines produced by an identify module over a desired image.
*************************************************************************

Outlines (in a special format produced by an identify module) can be
placed on any desired image (grayscale, color, or blank) and then this 
resulting image can be saved using the SaveImages module.

Settings:
Would you like to set the intensity (brightness) of the outlines to be
the same as the brightest point in the image, or the maximum possible
value for this image format?

If your image is quite dim, then putting bright white lines onto it may
not be useful. It may be preferable to make the outlines equal to the
maximal brightness already occurring in the image.

If you choose to display outlines on a Blank image, the maximum intensity
will default to 'Max possible'.

See also identify modules.
'''

    variable_revision_number = 1
    category = "Image Processing"
    
    def create_settings(self):
        self.module_name = 'OverlayOutlines'
        self.blank_image = cps.Binary("Do you want to display outlines on a blank image?",False)
        self.image_name = cps.ImageNameSubscriber("On which image would you like to display the outlines?","None")
        self.output_image_name = cps.ImageNameProvider("What do you want to call the image with the outlines displayed?")
        self.wants_color = cps.Choice("Do you want the output image to have color outlines or have outlines drawn on a grayscale image",
                                      [WANTS_COLOR, WANTS_GRAYSCALE])
        self.max_type = cps.Choice("Would you like the intensity (brightness) of the outlines to be the same as the brightest point in the image, or the maximum possible value for this image format? Note: if you chose to display on a Blank image, this will default to Max possible.",
                                   [MAX_IMAGE, MAX_POSSIBLE])
        self.outlines = []
        self.add_outline()
        self.add_outline_button = cps.DoSomething("Add another outline","Add", self.add_outline)

    def add_outline(self):
        class OutlineSettings(object):
            '''The settings for a single outline'''
            def __init__(self, outlines):
                self.key = uuid.uuid4()
                def remove(key = self.key, outlines = outlines):
                    index =  [x.key for x in outlines].index(key)
                    del outlines[index]
                
                self.outline_name = cps.OutlineNameSubscriber("What did you call the outlines that you would like to display?","None")
                default_color = (COLOR_ORDER[len(outlines)]
                                 if len(outlines) < len(COLOR_ORDER)
                                 else COLOR_ORDER[0])
                self.color = cps.Choice("What color do you want the outlines to be?",
                                        COLORS.keys(), default_color)
                self.remove_button = cps.DoSomething("Remove the above outline",
                                                     "Remove",
                                                     remove)
            
            def settings(self):
                return [self.outline_name, self.color]

            def visible_settings(self, is_color):
                if is_color:
                    return [self.outline_name, self.color, self.remove_button]
                else:
                    return [self.outline_name, self.remove_button]
        self.outlines.append(OutlineSettings(self.outlines))
    
    def prepare_to_set_values(self, setting_values):
        assert (len(setting_values) - 5) % 2 == 0
        self.outlines = []
        for i in range((len(setting_values) - 5)/2):
            self.add_outline()

    def settings(self):
        result = [self.blank_image, self.image_name, self.output_image_name,
                  self.wants_color, self.max_type]
        for outline in self.outlines:
            result += outline.settings()
        return result

    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        if from_matlab and variable_revision_number == 2:
            # Order is
            # image_name
            # outline name
            # max intensity
            # output_image_name
            # color
            setting_values = [cps.YES if setting_values[0]=="Blank" else cps.NO,
                              setting_values[0],
                              setting_values[3],
                              WANTS_COLOR,
                              setting_values[2],
                              setting_values[1],
                              setting_values[4]]
            from_matlab = False
            variable_revision_number = 1
        return setting_values, variable_revision_number, from_matlab

    def visible_settings(self):
        result = [self.blank_image]
        if not self.blank_image.value:
            result += [self.image_name]
        result += [self.output_image_name, self.wants_color]
        if self.wants_color.value == WANTS_GRAYSCALE:
            result += [self.max_type]
        for outline in self.outlines:
            result += outline.visible_settings(self.wants_color.value ==
                                               WANTS_COLOR)
        result += [self.add_outline_button]
        return result

    def run(self, workspace):
        if self.wants_color.value == WANTS_COLOR:
            pixel_data = self.run_color(workspace)
        else:
            pixel_data = self.run_bw(workspace)
        if self.blank_image.value:
            output_image = cpi.Image(pixel_data)
            workspace.image_set.add(self.output_image_name.value, output_image)
            if not workspace.frame is None:
                figure = workspace.create_or_find_figure(subplots=(1,1))
                if self.wants_color.value:
                    figure.subplot_imshow_color(0, 0, pixel_data, 
                                                self.output_image_name.value)
                else:
                    figure.subplot_imshow_bw(0, 0, pixel_data,
                                             self.output_image_name.value)
        else:
            image = workspace.image_set.get_image(self.image_name.value)
            output_image = cpi.Image(pixel_data,parent_image = image)
            workspace.image_set.add(self.output_image_name.value, output_image)
            if not workspace.frame is None:
                figure = workspace.create_or_find_figure(subplots=(2,1))
                if image.pixel_data.ndim == 2:
                    figure.subplot_imshow_bw(0,0,image.pixel_data,
                                             "Original: %s" %
                                             self.image_name.value)
                else:
                    figure.subplot_imshow_color(0,0,image.pixel_data,
                                                "Original: %s" %
                                                self.image_name.value,
                                                normalize=False)
                if self.wants_color.value:
                    figure.subplot_imshow_color(1, 0, pixel_data, 
                                                self.output_image_name.value)
                else:
                    figure.subplot_imshow_bw(1, 0, pixel_data,
                                             self.output_image_name.value)

    def run_bw(self, workspace):
        if self.blank_image.value:
            mask = workspace.get_outline(self.outlines[0].outline_name.value)
            pixel_data = np.zeros((mask.shape))
            maximum = 1
        else:
            image = workspace.image_set.get_image(self.image_name.value,
                                                  must_be_grayscale=True)
            pixel_data = image.pixel_data
            maximum = 1 if self.max_type == MAX_POSSIBLE else np.max(pixel_data)
            pixel_data = pixel_data.copy()
        for outline in self.outlines:
            mask = workspace.get_outline(outline.outline_name.value).astype(bool)
            pixel_data[mask] = maximum
        return pixel_data
    
    def run_color(self, workspace):
        if self.blank_image.value:
            mask = workspace.get_outline(self.outlines[0].outline_name.value)
            pixel_data = np.zeros((mask.shape[0],mask.shape[1],3))
        else:
            image = workspace.image_set.get_image(self.image_name.value)
            pixel_data = image.pixel_data
            if pixel_data.ndim == 2:
                pixel_data = np.dstack((pixel_data,pixel_data,pixel_data))
            else:
                pixel_data = pixel_data.copy()
        for outline in self.outlines:
            mask = workspace.get_outline(outline.outline_name.value).astype(bool)
            color = COLORS[outline.color.value]
            for i in range(3):
                pixel_data[:,:,i][mask] = float(color[i])/255.0
        return pixel_data
