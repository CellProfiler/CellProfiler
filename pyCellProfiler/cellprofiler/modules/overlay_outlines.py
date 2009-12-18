'''<b>Overlay outlines</b> places outlines produced by an identify module over a desired image.
<hr>

Outlines (in a special format produced by an identify module) can be
placed on any desired image (grayscale, color, or blank) and then this 
resulting image can be saved using the SaveImages module.

See also <b>IdentifyPrimAutomatic, IdentifySecondary, IdentifyTertiarySubregion</b>
'''

#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Developed by the Broad Institute
#Copyright 2003-2009
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org

__version__="$Revision$"

import numpy as np

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

    module_name = 'OverlayOutlines'
    variable_revision_number = 1
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
            On which image would you like to display the outlines?
            Choose the image to serve as the background for the outlines.
            You can choose from images that were loaded or created by modules
            previous to this one""")
        self.output_image_name = cps.ImageNameProvider(
            "Name the output image:",
            "OrigOverlay",
            doc="""
            What do you want to call the image with the outlines displayed?
            This setting names the overlay image. The name you provide will
            show up in image lists in later modules (for instance 
            <b>SaveImages</b>)""")
        self.wants_color = cps.Choice(
            "Select outline display mode:",
            [WANTS_COLOR, WANTS_GRAYSCALE], doc="""
            This option chooses how to display the outline contours around
            your objects. Color outlines produce a clearer display for
            images where the cell borders have a high intensity, but take
            up more space in memory. Grayscale outlines are displayed with
            either the highest possible intensity or the same intensity
            as the brightest pixel in the image.""")
        self.spacer = cps.Divider(line=False)
        self.max_type = cps.Choice(
            "Select method to determine brightness of outlines:",
            [MAX_IMAGE, MAX_POSSIBLE],
            doc = """Would you like the intensity (brightness) of the outlines to be "
            "the same as the brightest point in the image, or the maximum "
            "possible value for this image format?
            If your image is quite dim, then putting bright white lines
            onto it may not be useful. It may be preferable to make the
            outlines equal to the maximal brightness already occurring 
            in the image.""")
        self.outlines = []
        self.add_outline()
        self.add_outline_button = cps.DoSomething("", "Add outline", self.add_outline)

    def add_outline(self):
        group = cps.SettingsGroup()
        group.append("outline_name",
                     cps.OutlineNameSubscriber(
                "Select outlines to display:",
                "None", doc="""
                    Choose an outline from a previous <b>IdentifyPrimAutomatic</b>,
                    <b>IdentifySecondary</b> or <b>IdentifyTertiarySubregion</b>
                    module. Each of the Identify modules has a checkbox that
                    determines whether the outlines are saved. If you check this,
                    you'll be asked to supply a name for the outline; you
                    can then select that name here.
                    """))
        default_color = (COLOR_ORDER[len(self.outlines)]
                         if len(self.outlines) < len(COLOR_ORDER)
                         else COLOR_ORDER[0])
        group.append("color", cps.Choice(
                "Select outline color:",
                COLORS.keys(), default_color))
        group.append("remover", cps.RemoveSettingButton("", "Remove this outline", self.outlines, group))
        group.append("spacer", cps.Divider(line=False))
        self.outlines.append(group)

    def prepare_settings(self, setting_values):
        assert (len(setting_values) - 5) % 2 == 0
        self.outlines = []
        for i in range((len(setting_values) - 5)/2):
            self.add_outline()

    def settings(self):
        result = [self.blank_image, self.image_name, self.output_image_name,
                  self.wants_color, self.max_type]
        for outline in self.outlines:
            result += [outline.outline_name, outline.color]
        return result

    def visible_settings(self):
        result = [self.blank_image]
        if not self.blank_image.value:
            result += [self.image_name]
        result += [self.output_image_name, self.wants_color, self.spacer]
        if (self.wants_color.value == WANTS_GRAYSCALE and not
            self.blank_image.value):
            result += [self.max_type]
        for outline in self.outlines:
            if self.wants_color.value == WANTS_COLOR:
                result += outline.unpack_group()
            else:
                result += [outline.outline_name, outline.remover, outline.spacer]
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
                if self.wants_color.value == WANTS_COLOR:
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
                    pixel_data = (pixel_data * 255.0).astype(np.uint8)
                    figure.subplot_imshow_color(1, 0, pixel_data, 
                                                self.output_image_name.value,
                                                normalize=False)
                else:
                    figure.subplot_imshow_bw(1, 0, pixel_data,
                                             self.output_image_name.value)

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
            outline_image = image_set.get_image(
                self.outlines[0].outline_name.value,
                must_be_binary = True)
            mask = outline_image.pixel_data
            pixel_data = np.zeros((mask.shape[0],mask.shape[1],3))
        else:
            image = image_set.get_image(self.image_name.value)
            pixel_data = image.pixel_data
            if pixel_data.ndim == 2:
                pixel_data = np.dstack((pixel_data,pixel_data,pixel_data))
            else:
                pixel_data = pixel_data.copy()
        for outline in self.outlines:
            mask = image_set.get_image(outline.outline_name.value,
                                       must_be_binary=True).pixel_data
            color = COLORS[outline.color.value]
            for i in range(3):
                pixel_data[:,:,i][mask] = float(color[i])/255.0
        return pixel_data
    
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
        return setting_values, variable_revision_number, from_matlab

