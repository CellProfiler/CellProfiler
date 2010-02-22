'''<b>Tile</b> tiles images together to form one large image, either across image cycles or within a cycle
<hr>
This module allows many images to be viewed simultaneously in a grid layout you specify (e.g., in the actual layout in which the
images were collected).  If you want to view a large number of images, you will generate an extremely large file
(roughly the size of all the images' sizes added together).  This will cause memory errors, so we offer a few suggestions:
<ol>
<li>Resize the images to a fraction of their original size, using the <b>Resize</b> module just before this module in the pipeline.</li>
<li>Rescale the images to 8-bit, which decreases the number of graylevels in the image (thus decreasing resolution)
but also decreases the size of the image. </li>
<li> Use the <b>ConserveMemory</b> module just before this module to clear out images that are stored in memory. 
Place this module prior to the <b>Tile</b> module (and maybe also afterwards) and ask it to retain only those 
images which are needed for downstream modules. </li>
</ol>

This module replaces the functionality of the module <b>PlaceAdjacent</b>.
'''
__version__ = "$Revision: 9034 $"

import numpy as np
import scipy.ndimage as scind
import sys

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas

T_WITHIN_CYCLES = 'Within cycles'
T_ACROSS_CYCLES = 'Across cycles'
T_ALL = (T_WITHIN_CYCLES, T_ACROSS_CYCLES)

P_TOP_LEFT = 'top left'
P_BOTTOM_LEFT = 'bottom left'
P_TOP_RIGHT = 'top right'
P_BOTTOM_RIGHT = 'bottom right'
P_ALL = (P_TOP_LEFT, P_BOTTOM_LEFT, P_TOP_RIGHT, P_BOTTOM_RIGHT)

S_ROW = "row"
S_COL = "column"
S_ALL = (S_ROW, S_COL)

'''Module dictionary keyword for storing the # of images in the group when tiling'''
IMAGE_COUNT = "ImageCount"
'''Dictionary keyword for storing the current image number in the group'''
IMAGE_NUMBER = "ImageNumber"
'''Module dictionary keyword for the image being tiled'''
TILED_IMAGE = "TiledImage"
TILE_WIDTH = "TileWidth"
TILE_HEIGHT = "TileHeight"

FIXED_SETTING_COUNT = 10

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
             <li><i>Tile within cycles:</i> This option takes the place of the <b>PlaceAdjacent</b> module.  For example, 
             you may tile three different channels (OrigRed, OrigBlue, and OrigGreen), and a new tiled image will 
             be created for every image cycle.</li>
             <li><i>Tile across cycles:</i> This module replicates the original <b>Tile</b> functionality.  For example, 
             you may tile all the images you wish to analyze (OrigBlue), and one final tiled image will be created 
             when processing is complete.</li>
             </ul>''')
        self.rows = cps.Integer("Number of rows in final tiled image",
                                         8, doc='''How many rows would you like to have in the tiled image?
                                         For example, if you want to show your images in a 96-well format, 
                                         enter 8.''')
        self.columns = cps.Integer("Number of columns in final tiled image",
                                         12, doc='''How many columns would you like to have in the tiled image?
                                         For example, if you want to show your images in a 96-well format, 
                                         enter 12.''')
        self.place_first = cps.Choice("Begin tiling in this corner of the final image", P_ALL, doc = '''
            Where do you want the first image to be placed?  Begin in the upper left-hand corner
            for a typical multi-well plate format where the first image is A01.''')
        self.tile_style = cps.Choice("Begin tiling across a row, or down a column?", S_ALL, doc = '''
            Are the images arranged in rows or columns?  If your images are named A01, A02, etc, 
            enter "row".''')
        self.meander = cps.Binary("Tile in meander mode?", False, '''Meander mode tiles adjacent images in one direction, 
                                then the next row/column in the opposite direction.  Some microscopes capture images
                                in this fashion.''')
        self.wants_automatic_rows = cps.Binary(
            "Automatically calculate # of rows?", False,
            doc = """<b>Tile</b> can automatically calculate the number of rows
            in the grid from the number of image sets that will be processed.
            Check this box to create a grid that has the number of columns
            that you entered and enough rows to display all of your images.
            If you check both automatic rows and automatic columns, <b>Tile</b>
            will create a grid that has roughly the same number of rows
            and columns.
            """)
        self.wants_automatic_columns = cps.Binary(
            "Automatically calculate # of columns?", False,
            doc = """<b>Tile</b> can automatically calculate the number of columns
            in the grid from the number of image sets that will be processed.
            Check this box to create a grid that has the number of rows
            that you entered and enough columns to display all of your images.""")
        
    def add_image(self, can_remove = True):
        '''Add an image + associated questions and buttons'''
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=True))
        
        group.append("input_image_name", 
                     cps.ImageNameSubscriber("Select an additional image",
                                            "None",doc="""
                                            What is the name of the additional image to tile?"""))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove above image", self.additional_images, group))
        self.additional_images.append(group)

    def settings(self):
        result = [self.input_image, self.output_image, self.tile_method, 
                  self.rows, self.columns, self.place_first, self.tile_style, 
                  self.meander, self.wants_automatic_rows, 
                  self.wants_automatic_columns]
      
        for additional in self.additional_images:
            result += [additional.input_image_name]
        return result

    def prepare_settings(self, setting_values):
        assert (len(setting_values)-FIXED_SETTING_COUNT)% 1 == 0
        n_additional = (len(setting_values)-FIXED_SETTING_COUNT)/1
        del self.additional_images[:]
        while len(self.additional_images) < n_additional:
            self.add_image()

    def visible_settings(self):
        result = [self.input_image, self.output_image, self.tile_method,
                  self.wants_automatic_rows]
        if not self.wants_automatic_rows:
            result += [self.rows]
        result += [self.wants_automatic_columns]
        if not self.wants_automatic_columns:
            result += [self.columns]

        result += [self.place_first, self.tile_style, self.meander]
        
        if self.tile_method == T_WITHIN_CYCLES:
            for additional in self.additional_images:
                result += additional.visible_settings()
            result += [self.add_button]
        return result

    def prepare_group(self, pipeline, image_set_list, grouping, image_numbers):
        '''Prepare to handle a group of images when tiling'''
        d = self.get_dictionary(image_set_list)
        d[IMAGE_COUNT] = len(image_numbers)
        d[IMAGE_NUMBER] = 0
        d[TILED_IMAGE] = None
        
    def run(self, workspace):
        '''do the image analysis'''
        if self.tile_method == T_WITHIN_CYCLES:
            output_pixels = self.place_adjacent(workspace)
        else:
            output_pixels = self.tile(workspace)
        output_image = cpi.Image(output_pixels)
        workspace.image_set.add(self.output_image.value, output_image)
        if workspace.frame is not None:
            workspace.display_data.image = output_pixels
    
    def display(self, workspace):
        '''Display 
        '''
        figure = workspace.create_or_find_figure(subplots=(1,1))
        pixels = workspace.display_data.image
        name = self.output_image.value
        if pixels.ndim == 3:
            figure.subplot_imshow(0, 0, pixels, title = name)
        else:
            figure.subplot_imshow_grayscale(0, 0, pixels, title = name)

    def is_interactive(self):
        return False
        
    def tile(self, workspace):
        '''Tile images across image sets
        '''
        d = self.get_dictionary(workspace.image_set_list)
        rows, columns = self.get_grid_dimensions(d[IMAGE_COUNT])
        image_set = workspace.image_set
        assert isinstance(image_set, cpi.ImageSet)
        image = image_set.get_image(self.input_image)
        pixels = image.pixel_data
        if d[TILED_IMAGE] is None:
            tile_width = pixels.shape[1]
            tile_height = pixels.shape[0]
            height = tile_height * rows
            width = tile_width * columns
            if pixels.ndim == 3:
                shape = (height, width, pixels.shape[2])
            else:
                shape = (height, width)
            output_pixels = np.zeros(shape)
            d[TILED_IMAGE] = output_pixels
            d[TILE_WIDTH] = tile_width
            d[TILE_HEIGHT] = tile_height
        else:
            output_pixels = d[TILED_IMAGE]
            tile_width = d[TILE_WIDTH]
            tile_height = d[TILE_HEIGHT]
        
        image_index = d[IMAGE_NUMBER]
        d[IMAGE_NUMBER] = image_index + 1
        self.put_tile(pixels, output_pixels, image_index, rows, columns)
        return output_pixels
    
    def put_tile(self, pixels, output_pixels, image_index, rows, columns):
        tile_height = int(output_pixels.shape[0] / rows)
        tile_width = int(output_pixels.shape[1] / columns)
        tile_i, tile_j = self.get_tile_ij(image_index, rows, columns)
        tile_i *= tile_height
        tile_j *= tile_width
        img_height = min(tile_height, pixels.shape[0])
        img_width = min(tile_width, pixels.shape[1])
        if output_pixels.ndim == 2:
            output_pixels[tile_i:(tile_i + img_height),
                          tile_j:(tile_j + img_width)] = \
                pixels[:img_height, :img_width]
        elif pixels.ndim == 3:
            output_pixels[tile_i:(tile_i + img_height),
                          tile_j:(tile_j + img_width),:] = \
                pixels[:img_height, :img_width,:]
        else:
            for k in range(output_pixels.shape[2]):
                output_pixels[tile_i:(tile_i+img_height),
                              tile_j:(tile_j+img_width), k] = \
                             pixels[:img_height, :img_width]
        return output_pixels
    
    def place_adjacent(self, workspace):
        '''Place images from the same image set adjacent to each other'''
        rows, columns = self.get_grid_dimensions()
        image_names = ([ self.input_image.value ] +
                       [ g.input_image_name.value 
                         for g in self.additional_images])
        pixel_data = [workspace.image_set.get_image(name).pixel_data
                      for name in image_names]
        tile_width = 0
        tile_height = 0
        colors = 0
        for p in pixel_data:
            tile_width = max(tile_width, p.shape[1])
            tile_height = max(tile_height, p.shape[0])
            if p.ndim > 2:
                colors = 3
        height = tile_height * rows
        width = tile_width * columns
        if colors > 0:
            output_pixels = np.zeros((height, width, colors))
        else:
            output_pixels = np.zeros((height, width))
        for i, p in enumerate(pixel_data):
            self.put_tile(p, output_pixels, i, rows, columns)
        return output_pixels
    
    def get_tile_ij(self, image_index, rows, columns):
        '''Get the I/J coordinates for an image
        
        returns i,j where 0 < i < self.rows and 0 < j < self.columns
        '''
        if self.tile_style == S_ROW:
            tile_i = int(image_index / columns)
            tile_j = image_index % columns
            if self.meander and tile_i % 2 == 1:
                # Reverse the direction if in meander mode
                tile_j = columns - tile_j - 1
        else:
            tile_i = image_index % rows
            tile_j = int(image_index / rows)
            if self.meander and tile_j % 2 == 1:
                # Reverse the direction if in meander mode
                tile_i = rows - tile_i - 1
        if self.place_first in (P_BOTTOM_LEFT, P_BOTTOM_RIGHT):
            tile_i = rows - tile_i - 1
        if self.place_first in (P_TOP_RIGHT, P_BOTTOM_RIGHT):
            tile_j = columns - tile_j - 1
        if (tile_i < 0 or tile_i >= rows or
            tile_j < 0 or tile_j >= columns):
            raise ValueError(("The current image falls outside of the grid boundaries. \n"
                              "Grid dimensions: %d, %d\n" 
                              "Tile location: %d, %d\n") %
                             (columns, rows,
                              tile_j, tile_i))
        return tile_i, tile_j
    
    def get_grid_dimensions(self, image_count = None):
        '''Get the dimensions of the grid in i,j format
        
        image_count - # of images in the grid. If None, use info from settings.
        '''
        assert ((image_count is not None) or 
                self.tile_method == T_WITHIN_CYCLES), "Must specify image count for %s method" % self.tile_method.value
        if image_count is None:
            image_count = len(self.additional_images) + 1
        if self.wants_automatic_rows:
            if self.wants_automatic_columns:
                #
                # Take the square root of the # of images & assign as rows.
                # Maybe add 1 to get # of columns.
                #
                i = int(np.sqrt(image_count))
                j = int((image_count + i - 1) / i)
                return i,j
            else:
                j = self.columns.value
                i = int((image_count + j - 1) / j)
                return i,j
        elif self.wants_automatic_columns:
            i = self.rows.value
            j = int((image_count + i - 1) / i)
            return i,j
        else:
            return self.rows.value, self.columns.value
        
    def get_measurement_columns(self, pipeline):
        '''return the measurements'''
        columns = []
        return columns
    
    def validate_module(self, pipeline):
        '''Make sure the settings are consistent
        
        Check to make sure that we have enough rows and columns if
        we are in PlaceAdjacent mode.
        '''
        if (self.tile_method == T_WITHIN_CYCLES and
            (not self.wants_automatic_rows) and 
            (not self.wants_automatic_columns) and
            self.rows.value * self.columns.value <
            len(self.additional_images) + 1):
            raise cps.ValidationError(
                "There are too many images (%d) for a %d by %d grid" %
                (len(self.additional_images)+1, self.columns.value, 
                 self.rows.value),
                self.rows)

    def upgrade_settings(self, setting_values, 
                         variable_revision_number, 
                         module_name, from_matlab):
        '''this must take into account both Tile and PlaceAdjacent from the Matlab'''
        if (from_matlab and module_name == "Tile" and 
            variable_revision_number == 1):
            image_name, orig_image_name, tiled_image, number_rows,\
            number_columns, row_or_column, top_or_bottom, left_or_right,\
            meander_mode, size_change = setting_values
            
            if size_change != "1":
                sys.stderr.write(
                    "Discarding rescaling during import of Tile. "
                    "Use the resize module with a factor of %s.\n" % 
                    size_change)
                
            left = left_or_right.lower() == 'left'
            if top_or_bottom.lower() == 'top':
                place_first = P_TOP_LEFT if left else P_TOP_RIGHT
            else:
                place_first = P_BOTTOM_LEFT if left else P_BOTTOM_RIGHT
                               
            tile_style = S_ROW if row_or_column.lower() == 'row' else S_COL
            
            wants_automatic_rows = cps.NO
            wants_automatic_columns = cps.NO
            if number_rows == cps.AUTOMATIC:
                number_rows = 8
                wants_automatic_rows = cps.YES
            if number_columns == cps.AUTOMATIC:
                number_columns = 12
                wants_automatic_columns = cps.YES
            setting_values = [ image_name, tiled_image, T_ACROSS_CYCLES,
                               number_rows, number_columns, place_first,
                               tile_style, meander_mode, wants_automatic_rows,
                               wants_automatic_columns]
            from_matlab = False
            variable_revision_number = 1
            module_name = self.module_name
            
        if (from_matlab and module_name == "PlaceAdjacent" and
            variable_revision_number == 3):
            image_names = [s for s in setting_values[:6]
                           if s.lower() != cps.DO_NOT_USE.lower()]
            adjacent_image_name = setting_values[6]
            horizontal_or_vertical = setting_values[7]
            delete_pipeline = setting_values[8]
            if delete_pipeline == cps.YES:
                sys.stderr.write(
                    "Ignoring memory option when importing PlaceAdjacent "
                    "into Tile. Use the ConserveMemory module to remove "
                    "the image from memory if desired.\n")
            if len(image_names) == 0:
                image_names.append(cps.DO_NOT_USE)
            if horizontal_or_vertical.lower() == "horizontal":
                tile_style = S_ROW
                number_rows = "1"
                number_columns = str(len(image_names))
            else:
                tile_style = S_COL
                number_rows = str(len(image_names))
                number_columns = "1"
                
            setting_values = [image_names[0], adjacent_image_name,
                              T_WITHIN_CYCLES, number_rows, number_columns,
                              P_TOP_LEFT, tile_style, cps.NO, cps.NO, cps.NO]
            setting_values += image_names[1:]
            variable_revision_number = 1
            from_matlab = False
            module_name = self.module_name
            
        return setting_values, variable_revision_number, from_matlab
