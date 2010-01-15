'''<b>Tile</b> Tiles images together to form one large image, either across image cycles or within a cycle
<hr>
Allows many images to be viewed simultaneously in a grid layout you specify (e.g., in the actual layout in which the
images were collected).  If you want to view a large number of images, you will generate an extremely large file
(roughly the size of all the images' sizes added together).  This will cause memory errors, so we offer a few suggestions:
<ol>
<li>Resize the images to a fraction of their original size by using the Resize module just before this module.</li>
<li>Rescale the images to 8-bit, which decreases the number of graylevels in the image (thus decreasing resolution)
but also decreases the size of the image. </li>
<li> Use the SpeedUpCellProfiler module just before this module to clear out images that are stored in memory. 
Place this module just prior to the Tile module (and maybe also afterwards) and ask it to retain only those 
images which are needed for downstream modules. </li>
</ol>

This module combines the functionality of both Tile and PlaceAdjacent.
'''
__version__ = "$Revision: 9034 $"

import numpy as np
import scipy.ndimage as scind

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas

T_WITHIN_CYCLES = 'Within cycles'
T_ACROSS_CYCLES = 'Across cycles'
T_ALL = (T_WITHIN_CYCLES, T_ACROSS_CYCLES)

P_TOP_LEFT = 'top left'
P_BOTT_RIGHT = 'bottom left'
P_TOP_RIGHT = 'top right'
P_BOTT_RIGHT = 'bottom right'
P_ALL = (P_TOP_LEFT, P_BOTT_RIGHT, P_TOP_RIGHT, P_BOTT_RIGHT)

S_ROW = "row"
S_COL = "column"
S_ALL = (S_ROW, S_COL)


class Tile(cpm.CPModule):
    module_name = "Tile"
    category = 'Image Processing'
    variable_revision_number = 1

    def create_settings(self):
        self.input_image = cps.ImageNameSubscriber("Select the input image",
                                                         "None",doc="""
                                                         What did you call the image to be tiled?""")
        self.output_image = cps.ImageNameProvider("Name the output image",
                                                        "TiledImage",doc="""
                                                        What do you want to call the final tiled image?""")
        self.additional_images = []
        self.add_button = cps.DoSomething("", "Add another image",
                                          self.add_image)
        self.tile_method = cps.Choice("Tile within cycles or across cycles?",
                                           T_ALL, doc='''
             How would you like to tile images? Two options are available:<br>
             <ul>
             <li><i>Tile within cycles:</i> This option takes the place of the PlaceAdjacent module.  For example, 
             you may tile three different channels (OrigRed, OrigBlue, and OrigGreen), and a new tiled image will 
             be created for every image cycle.</li>
             <li><i>Tile across cycles:</i> This module replicates the original Tile functionality.  For example, 
             you may tile all the images you wish to analyze (OrigBlue), and one final tiled image will be created 
             when processing is complete.</li>
             </ul>''')
        self.rows = cps.Integer("Number of rows in final tiled image:",
                                         8, doc='''How many rows would you like to have in the tiled image?
                                         For example, if you want to show your images in a 96-well format, you would
                                         enter 8.''')
        self.columns = cps.Integer("Number of columns in final tiled image:",
                                         12, doc='''How many columns would you like to have in the tiled image?
                                         For example, if you want to show your images in a 96-well format, you would
                                         enter 12.''')
        self.place_first = cps.Choice("Begin tiling in this corner of the final image:", P_ALL, doc = '''
            Where do you want the first image to be placed?  You would begin in the upper left-hand corner
            for a typical multi-well plate format where the first image is A01.''')
        self.tile_style = cps.Choice("Begin tiling across a row, or down a column?", S_ALL, doc = '''
            Are the images arranged in rows or columns?  If your images are named A01, A02, etc, you would
            enter "row".''')
        self.meander = cps.Binary("Tile in meander mode?", False, '''Meander mode tiles adjacent images in one direction, 
                                then the next row/column in the opposite direction.  Some microscopes capture images
                                in this fashion.''')

        
    def add_image(self, can_remove = True):
        '''Add an image + associated questions and buttons'''
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=True))
        
        group.append("input_image_name", 
                     cps.ImageNameSubscriber("Select an additional image:",
                                            "None",doc="""
                                            What is the name of the additional image to tile?"""))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove above image", self.additional_images, group))
        self.additional_images.append(group)

    def settings(self):
        result = [self.input_image, self.output_image, self.tile_method, self.rows, self.columns,
                  self.place_first, self.tile_style, self.meander]
      
        for additional in self.additional_images:
            result += [additional.input_image_name]
        return result

    def prepare_settings(self, setting_values):
        assert (len(setting_values)-8)% 1 == 0
        n_additional = (len(setting_values)-8)/1
        del self.additional_images[:]
        while len(self.additional_images) < n_additional:
            self.add_image()

    def visible_settings(self):
        result = [self.input_image, self.output_image, self.tile_method, self.rows, self.columns,
                  self.place_first, self.tile_style, self.meander]
        
        for additional in self.additional_images:
            result += additional.visible_settings()
        result += [self.add_button]
        return result

    def run(self, workspace):
        '''do the image analysis'''
        
    
    def display(self, workspace):
        '''Display 
        '''

    def is_interactive(self):
        return False
        
    def tile(self, workspace):
        '''tile the stuffs
        '''
    def get_measurement_columns(self, pipeline):
        '''return the measurements'''
        return columns

    def upgrade_settings(self, setting_values, 
                         variable_revision_number, 
                         module_name, from_matlab):
        '''this must take into account both Tile and PlaceAdjacent from the Matlab'''
        
        return setting_values, variable_revision_number, from_matlab