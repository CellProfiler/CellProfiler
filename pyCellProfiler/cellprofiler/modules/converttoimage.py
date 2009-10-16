'''converttoimage.py - the ConvertToImage module

Converts a labels matrix into an image

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

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

class ConvertToImage(cpm.CPModule):
    '''SHORT DESCRIPTION:
    Converts objects you have identified into an image so that it can be
    saved with the Save Images module.
    *************************************************************************
    
    This module allows you to take previously identified objects and convert
    them into an image, which can then be saved with the SaveImages modules.
    
    Settings:
    
    Binary (black & white), grayscale, or color: Choose how you would like
    the objects to appear. Color allows you to choose a colormap which will
    produce jumbled colors for your objects. Grayscale will give each object
    a graylevel pixel intensity value corresponding to its number (also
    called label), so it usually results in objects on the left side of the
    image being very dark, and progressing towards white on the right side of
    the image. You can choose "Color" with a "Gray" colormap to produce
    jumbled gray objects.
    
    Colormap:
    Affect how the objects are colored. You can look up your default colormap
    under File > Set Preferences. Look in matlab help online (try Google) to
    see what the available colormaps look like. See also Help > HelpColormaps
    in the main CellProfiler window.
    '''
    
    category = "Object Processing"
    variable_revision_number = 1
    
    def create_settings(self):
        self.module_name = "ConvertToImage"
        self.object_name = cps.ObjectNameSubscriber("What did you call the objects you want to convert to an image?","None")
        self.image_name = cps.ImageNameProvider("What do you want to call the resulting image?", "CellImage")
        self.image_mode = cps.Choice("What colors should the resulting image use?",
                                     IM_ALL)
        self.colormap = cps.Choice("What do you want the colormap to be?",
                                   COLORMAPS)


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
            if not workspace.frame is None:
                figure.subplot_imshow_color(1,0,pixel_data, self.image_name.value)
        elif self.image_mode == IM_UINT16:
            pixel_data = labels.copy()
            if not workspace.frame is None:
                figure.subplot_imshow_grayscale(1,0,pixel_data,
                                                self.image_name.value)
            convert = False
        image = cpi.Image(pixel_data, parent_image = objects.parent_image,
                          convert = convert)
        workspace.image_set.add(self.image_name.value, image)
    
    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        if variable_revision_number == 1 and from_matlab:
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab


