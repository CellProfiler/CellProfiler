'''<b>Convert Objects To Image </b> converts objects you have identified into an image
<hr>
This module allows you to take previously identified objects and convert
them into an image according to a colormap you select, which can then be saved with the <b>SaveImages</b> modules.
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
import matplotlib.cm

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi
import cellprofiler.preferences as cpprefs
from cellprofiler.gui.cpfigure import renumber_labels_for_display

DEFAULT_COLORMAP = "Default"
COLORCUBE = "colorcube"
LINES = "lines"
WHITE = "white"
COLORMAPS = ["Default", "autumn", "bone", COLORCUBE, "cool", "copper",
             "flag", "gray", "hot", "hsv", "jet", LINES,"pink", "prism",
             "spring", "summer", WHITE, "winter" ]

IM_COLOR = "Color"
IM_BINARY = "Binary (black & white)"
IM_GRAYSCALE = "Grayscale"
IM_UINT16 = "uint16"
IM_ALL = [IM_COLOR, IM_BINARY, IM_GRAYSCALE, IM_UINT16]

class ConvertObjectsToImage(cpm.CPModule):

    module_name = "ConvertObjectsToImage"
    category = "Object Processing"
    variable_revision_number = 1
    
    def create_settings(self):
        self.object_name = cps.ObjectNameSubscriber("Select the input objects","None",doc="""
                                What did you call the objects you want to convert to an image?""")
        
        self.image_name = cps.ImageNameProvider("Name the output image", "CellImage",doc="""
                                What do you want to call the resulting image?""")
        
        self.image_mode = cps.Choice("Select the color type",
                                IM_ALL,doc="""
                                What colors should the resulting image use? Choose how you would like
                                the objects to appear:
                                <ul>
                                <li><i>Color:</i> Allows you to choose a colormap that will
                                produce jumbled colors for your objects. </li>
                                <li><i>Grayscale:</i> Gives each object
                                a graylevel pixel intensity value corresponding to its number (also
                                called label), so it usually results in objects on the left side of the
                                image being very dark, progressing toward white on the right side of
                                the image. </li>
                                <li><i>uint16:</i> Gives each object a different number,
                                from 1 to 65535 (the numbers that you can put in
                                a 16-bit integer) and numbers all pixels in each
                                object with the object's number. This format can
                                be written out as a .MAT or .TIFF file if you
                                want to process the label matrix image using
                                another program.</li>
                                </ul>
                                You can choose <i>Color</i> with a <i>Gray</i> colormap to produce
                                jumbled gray objects.""")
        
        self.colormap = cps.Choice("Select the colormap",
                                COLORMAPS,doc="""
                                What do you want the colormap to be? This setting affects how the objects are colored. 
                                You can look up your default colormap under <i>File > Set Preferences</i>.""")

    def settings(self):
        return [self.object_name, self.image_name, self.image_mode, 
                self.colormap]

    def visible_settings(self):
        if self.image_mode == IM_COLOR:
            return [self.object_name, self.image_name, self.image_mode, 
                    self.colormap]
        else:
            return [self.object_name, self.image_name, self.image_mode] 

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.object_name.value)
        labels = objects.segmented
        convert = True
        if not workspace.frame is None:
            figure = workspace.create_or_find_figure(subplots=(2,1))
            figure.subplot_imshow_labels(0,0,labels,
                                         "Original: %s"%self.object_name.value)
        if self.image_mode == IM_BINARY:
            pixel_data = labels != 0
            if not workspace.frame is None:
                figure.subplot_imshow_bw(1,0,pixel_data,self.image_name.value)
        elif self.image_mode == IM_GRAYSCALE:
            pixel_data = labels.astype(float) / np.max(labels)
            if not workspace.frame is None:
                figure.subplot_imshow_grayscale(1,0,pixel_data,self.image_name.value)
        elif self.image_mode == IM_COLOR:
            if self.colormap.value == DEFAULT_COLORMAP:
                cm_name = cpprefs.get_default_colormap()
            elif self.colormap.value == COLORCUBE:
                # Colorcube missing from matplotlib
                cm_name = "gist_rainbow"
            elif self.colormap.value == LINES:
                # Lines missing from matplotlib and not much like it,
                # Pretty boring palette anyway, hence
                cm_name = "Pastel1"
            elif self.colormap.value == WHITE:
                # White missing from matplotlib, it's just a colormap
                # of all completely white... not even different kinds of
                # white. And, isn't white just a uniform sampling of
                # frequencies from the spectrum?
                cm_name = "Spectral"
            else:
                cm_name = self.colormap.value
            cm = matplotlib.cm.get_cmap(cm_name)
            mapper = matplotlib.cm.ScalarMappable(cmap=cm)
            pixel_data = mapper.to_rgba(renumber_labels_for_display(labels))
            pixel_data = pixel_data[:,:,:3]
            pixel_data[labels == 0,:] = 0
            if not workspace.frame is None:
                figure.subplot_imshow(1,0,pixel_data, self.image_name.value)
        elif self.image_mode == IM_UINT16:
            pixel_data = labels.copy()
            if not workspace.frame is None:
                figure.subplot_imshow_grayscale(1,0,pixel_data,
                                                self.image_name.value)
            convert = False
        image = cpi.Image(pixel_data, parent_image = objects.parent_image,
                          convert = convert)
        workspace.image_set.add(self.image_name.value, image)
    
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        if variable_revision_number == 1 and from_matlab:
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab

#
# Backwards compatability
#
ConvertToImage = ConvertObjectsToImage
