'''<b>Overlay Outlines</b> places outlines produced by an <b>Identify</b> module over a desired image
<hr>

This module places outlines (in a special format produced by an <b>Identify</b> module) on any desired image (grayscale, color, or blank). The 
resulting image can be saved using the <b>SaveImages</b> module.

See also <b>IdentifyPrimaryObjects, IdentifySecondaryObjects, IdentifyTertiaryObjects</b>.
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

import numpy as np
from scipy.ndimage import distance_transform_edt

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps

WANTS_COLOR = "Color"
WANTS_GRAYSCALE = "Grayscale"

MAX_IMAGE = "Max of image"
MAX_POSSIBLE = "Max possible"

COLORS = { "White":  (1,1,1),
           "Black":  (0,0,0),
           "Red":    (1,0,0),
           "Green":  (0,1,0),
           "Blue":   (0,0,1),
           "Yellow": (1,1,0) }

COLOR_ORDER = ["Red", "Green", "Blue","Yellow","White","Black"]

class OverlayOutlines(cpm.CPModule):

    module_name = 'OverlayOutlines'
    variable_revision_number = 2
    category = "Image Processing"
    
    def create_settings(self):
        self.blank_image = cps.Binary("Display outlines on a blank image?",
                                      False, doc="""
                        If you check this setting, the module will produce an
                        image of the outlines on a black background. If the
                        setting is unchecked, the module will overlay the 
                        outlines on an image of your choosing.""")
        self.image_name = cps.ImageNameSubscriber(
            "Select image on which to display outlines","None", doc="""
            <i>(Used only when a blank image has not been selected)</i> <br>
            On which image would you like to display the outlines?
            Choose the image to serve as the background for the outlines.
            You can choose from images that were loaded or created by modules
            previous to this one.""")
        self.line_width = cps.Float(
            "Width of outlines", "1",
            doc = """Enter the width, in pixels, of the
            outlines to be displayed on the image.""")
        self.output_image_name = cps.ImageNameProvider(
            "Name the output image",
            "OrigOverlay",
            doc="""
            What do you want to call the image with the outlines displayed?
            This will be the name of the overlay image, which you can 
            select in later modules (for instance, <b>SaveImages</b>).""")
        self.wants_color = cps.Choice(
            "Select outline display mode",
            [WANTS_COLOR, WANTS_GRAYSCALE], doc="""
            Specify how to display the outline contours around
            your objects. Color outlines produce a clearer display for
            images where the cell borders have a high intensity, but take
            up more space in memory. Grayscale outlines are displayed with
            either the highest possible intensity or the same intensity
            as the brightest pixel in the image.""")
        self.spacer = cps.Divider(line=False)
        self.max_type = cps.Choice(
            "Select method to determine brightness of outlines",
            [MAX_IMAGE, MAX_POSSIBLE],
            doc = """
            <i>(Used only when outline display mode is grayscale)</i> <br>
            Would you like the intensity (brightness) of the outlines to be 
            the same as the brightest point in the image, or the maximum 
            possible value for this image format?
            If your image is quite dim, then putting bright white lines
            onto it may not be useful. It may be preferable to make the
            outlines equal to the maximal brightness already occurring 
            in the image.""")
        self.outlines = []
        self.add_outline(can_remove = False)
        self.add_outline_button = cps.DoSomething("", "Add another outline", self.add_outline)

    def add_outline(self,can_remove = True):
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))
            
        group.append("outline_name",
                     cps.OutlineNameSubscriber(
                "Select outlines to display",
                "None", doc="""
                    Choose outlines to display, from a previous <b>Identify</b>
                    module. Each of the <b>Identify</b> modules has a checkbox that
                    determines whether the outlines are saved. If you have checked this,
                    you were asked to supply a name for the outline; you
                    can then select that name here.
                    """))
        default_color = (COLOR_ORDER[len(self.outlines)]
                         if len(self.outlines) < len(COLOR_ORDER)
                         else COLOR_ORDER[0])
        group.append("color", cps.Choice(
                "Select outline color",
                COLORS.keys(), default_color))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove this outline", self.outlines, group))
        
        self.outlines.append(group)

    def prepare_settings(self, setting_values):
        assert (len(setting_values) - 6) % 2 == 0
        self.outlines = []
        for i in range((len(setting_values) - 6)/2):
            self.add_outline()

    def settings(self):
        result = [self.blank_image, self.image_name, self.output_image_name,
                  self.wants_color, self.max_type, self.line_width]
        for outline in self.outlines:
            result += [outline.outline_name, outline.color]
        return result

    def visible_settings(self):
        result = [self.blank_image]
        if not self.blank_image.value:
            result += [self.image_name]
        result += [self.output_image_name, self.wants_color, 
                   self.line_width, self.spacer]
        if (self.wants_color.value == WANTS_GRAYSCALE and not
            self.blank_image.value):
            result += [self.max_type]
        for outline in self.outlines:
            if self.wants_color.value == WANTS_COLOR:
                result += outline.visible_settings()
            else:
                result += [outline.outline_name]
                if hasattr(outline, "remover"):
                    result += [outline.remover]
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
                figure = workspace.create_or_find_figure(title="OverlayOutlines, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(1,1))
                if self.wants_color.value == WANTS_COLOR:
                    figure.subplot_imshow(0, 0, pixel_data, 
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
                    figure.subplot_imshow_bw(0, 0, image.pixel_data,
                                             "Original: %s" %
                                             self.image_name.value)
                else:
                    figure.subplot_imshow_color(0, 0, image.pixel_data,
                                          "Original: %s" %
                                          self.image_name.value,
                                          normalize=False)
                if self.wants_color.value == WANTS_COLOR:
                    figure.subplot_imshow(1, 0, pixel_data,
                                          self.output_image_name.value,
                                          sharex = figure.subplot(0,0),
                                          sharey = figure.subplot(0,0))
                else:
                    figure.subplot_imshow_bw(1, 0, pixel_data,
                                             self.output_image_name.value,
                                             sharex = figure.subplot(0,0),
                                             sharey = figure.subplot(0,0))

    def run_bw(self, workspace):
        image_set = workspace.image_set
        assert isinstance(image_set, cpi.ImageSet)
        if self.blank_image.value:
            outline_image = image_set.get_image(
                self.outlines[0].outline_name.value,
                must_be_binary = True)
            mask = outline_image.pixel_data
            pixel_data = np.zeros((mask.shape))
            maximum = 1
        else:
            image = image_set.get_image(self.image_name.value,
                                        must_be_grayscale=True)
            pixel_data = image.pixel_data
            maximum = 1 if self.max_type == MAX_POSSIBLE else np.max(pixel_data)
            pixel_data = pixel_data.copy()
        for outline in self.outlines:
            mask = image_set.get_image(outline.outline_name.value,
                                       must_be_binary=True).pixel_data
            pixel_data[mask] = maximum
        return pixel_data
    
    def run_color(self, workspace):
        image_set = workspace.image_set
        if self.blank_image.value:
            pixel_data = None
            pdmax = 1
        else:
            image = image_set.get_image(self.image_name.value)
            pixel_data = image.pixel_data
            if pixel_data.ndim == 2:
                pixel_data = np.dstack((pixel_data,pixel_data,pixel_data))
            else:
                pixel_data = pixel_data.copy()
            pdmax = float(np.max(pixel_data))
            if pdmax <= 0:
                pdmax = 1
        for outline in self.outlines:
            color = COLORS[outline.color.value]
            outline_img = self.get_outline(image_set, 
                                           outline.outline_name.value,
                                           color)
            if pixel_data is None:
                pixel_data = np.zeros(list(outline_img.shape[:2]) + [3], np.float32)
            i_max = min(outline_img.shape[0], pixel_data.shape[0])
            j_max = min(outline_img.shape[1], pixel_data.shape[1])
            outline_img = outline_img[:i_max, :j_max,:]
            window = pixel_data[:i_max, :j_max, :]
            alpha = outline_img[:,:,3]
            pixel_data[:i_max, :j_max, :] = (
                window * (1 - alpha[:,:,np.newaxis]) + 
                outline_img[:,:,:3] * alpha[:,:,np.newaxis] * pdmax)
            
        return pixel_data
    
    def get_outline(self, image_set, name, color):
        '''Get outline, with aliasing and taking widths into account'''
        pixel_data = image_set.get_image(name).pixel_data
        if pixel_data.ndim == 2:
            if len(color) == 3:
                color = np.hstack((color, [1]))
            pixel_data = pixel_data > 0
            output_image = color[np.newaxis, np.newaxis, :] * pixel_data[:,:,np.newaxis]
        else:
            output_image = np.dstack([pixel_data[:,:,i] for i in range(3)] +
                                     [np.sum(pixel_data, 2) > 0])
        if self.line_width.value > 1:
            half_line_width = float(self.line_width.value) / 2
            d, (i,j) = distance_transform_edt(output_image[:,:,3] == 0, 
                                              return_indices = True)
            mask = (d > 0) & (d <= half_line_width - .5)
            output_image[mask,:] = output_image[i[mask], j[mask],:]
            #
            # Do a little aliasing here using an alpha channel
            #
            mask = ((d > max(0, half_line_width - .5)) & 
                     (d < half_line_width + .5))
            d = half_line_width + .5 - d
            output_image[mask,:3] = output_image[i[mask], j[mask],:3]
            output_image[mask, 3] = d[mask]
            
        return output_image
    
    def upgrade_settings(self, setting_values, variable_revision_number, 
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
        if (not from_matlab) and variable_revision_number == 1:
            #
            # Added line width
            #
            setting_values = setting_values[:5] + ["1"] + setting_values[5:]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

