'''<b>Save Images </b> saves image or movie files
<hr>

Because CellProfiler usually performs many image analysis steps on many
groups of images, it does <i>not</i> save any of the resulting images to the
hard drive unless you specifically choose to do so with the <b>SaveImages</b> 
module. You can save any of the
processed images created by CellProfiler during the analysis using this module.

<p>You can choose from many different image formats for saving your files. This
allows you to use the module as a file format converter, by loading files
in their original format and then saving them in an alternate format.

<p>Note that saving images in 12- or 16-bit format is not supported.
<p>
See also <b>LoadImages</b>, <b>ConserveMemory</b>.
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

import matplotlib
import numpy as np
import os
import sys
import Image as PILImage
import scipy.io.matlab.mio
import traceback
try:
    from bioformats.formatreader import *
    from bioformats.formatwriter import *
    from bioformats.metadatatools import *
    has_bioformats = True
except:
    traceback.print_exc()
    sys.stderr.write(
        "Failed to load bioformats. SaveImages will not be able to save 16-bit TIFFS or movies.\n")
    has_bioformats = False
    
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements
import cellprofiler.settings as cps
import cellprofiler.preferences as cpp
from cellprofiler.gui.help import USING_METADATA_TAGS_REF, USING_METADATA_HELP_REF
from cellprofiler.preferences import \
     standardize_default_folder_names, DEFAULT_INPUT_FOLDER_NAME, \
     DEFAULT_OUTPUT_FOLDER_NAME, ABSOLUTE_FOLDER_NAME, \
     DEFAULT_INPUT_SUBFOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME, \
     IO_FOLDER_CHOICE_HELP_TEXT, IO_WITH_METADATA_HELP_TEXT

IF_IMAGE       = "Image"
IF_MASK        = "Mask"
IF_CROPPING    = "Cropping"
IF_FIGURE      = "Module window"
IF_MOVIE       = "Movie"
if has_bioformats:
    IF_ALL = [IF_IMAGE, IF_MASK, IF_CROPPING, IF_MOVIE, IF_FIGURE]
else:
    IF_ALL = [IF_IMAGE, IF_MASK, IF_CROPPING, IF_FIGURE]
    

FN_FROM_IMAGE  = "From image filename"
FN_SEQUENTIAL  = "Sequential numbers"
FN_SINGLE_NAME = "Single name"
SINGLE_NAME_TEXT = "Enter single file name"
FN_WITH_METADATA = "Name with metadata"
FN_IMAGE_FILENAME_WITH_METADATA = "Image filename with metadata"
METADATA_NAME_TEXT = ("""Enter file name with metadata""")
SEQUENTIAL_NUMBER_TEXT = "Enter file prefix"
FF_BMP         = "bmp"
FF_GIF         = "gif"
FF_HDF         = "hdf"
FF_JPG         = "jpg"
FF_JPEG        = "jpeg"
FF_PBM         = "pbm"
FF_PCX         = "pcx"
FF_PGM         = "pgm"
FF_PNG         = "png"
FF_PNM         = "pnm"
FF_PPM         = "ppm"
FF_RAS         = "ras"
FF_TIF         = "tif"
FF_TIFF        = "tiff"
FF_XWD         = "xwd"
FF_AVI         = "avi"
FF_MAT         = "mat"
FF_MOV         = "mov"
FF_SUPPORTING_16_BIT = [FF_TIF, FF_TIFF]
PC_WITH_IMAGE  = "Same folder as image"
OLD_PC_WITH_IMAGE_VALUES = ["Same folder as image"]
PC_CUSTOM      = "Custom"
PC_WITH_METADATA = "Custom with metadata"
WS_EVERY_CYCLE = "Every cycle"
WS_FIRST_CYCLE = "First cycle"
WS_LAST_CYCLE  = "Last cycle"
CM_GRAY        = "gray"

class SaveImages(cpm.CPModule):

    module_name = "SaveImages"
    variable_revision_number = 5
    category = "File Processing"
    
    def create_settings(self):
        self.save_image_or_figure = cps.Choice("Select the type of image to save",
                                               IF_ALL,
                                               IF_IMAGE,doc="""
                The following types of images can be saved as a file on the hard drive:
                <ul>
                <li><i>Image:</i> Any of the images produced upstream of <b>SaveImages</b> can be selected for saving. 
                Outlines created by <b>Identify</b> modules can also be saved with this option, but you must 
                select "Retain outlines..." of identified objects within the <b>Identify</b> module. You might
                also want to use the <b>OverlayOutlines</b> module prior to saving images.</li>
                <li><i>Crop mask (Relevant only if the Crop module is used):</i> The <b>Crop</b> module 
                creates a mask of the pixels of interest in the image. Saving the mask will produce a 
                binary image in which the pixels of interest are set to 1; all other pixels are 
                set to 0.</li>
                <li><i>Image's cropping (Relevant only if the Crop module is used):</i> The <b>Crop</b> 
                module also creates a cropping image which is typically the same size as the original 
                image. However, since the <b>Crop</b> permits removal of the rows and columns that are left 
                blank, the cropping can be of a different size than the mask.</li>
                <li><i>Movie:</i> A sequence of images can be saved as a movie file. Each 
                image becomes a frame of the movie.</li>
                <li><i>Module display window:</i> The window associated with a module can be saved, which
                will include all the panels and text within that window. <b>Currently, this option is not yet available.</b></li>
                </ul>
                Note that objects cannot be directly saved with the <b>SaveImages</b> module.
                You must first use the <b>ConvertObjectsToImage</b> module to convert the objects to an image, 
                followed by <b>SaveImages</b>.""")
        
        self.image_name  = cps.ImageNameSubscriber("Select the image to save","None", doc = """
                <i>(Used only if saving images, crop masks, and image croppings)</i><br>
                What did you call the images you want to save?""")
        
        self.figure_name = cps.FigureSubscriber("Select the module display window to save","None",doc="""
                <i>(Used only if saving module display windows)</i><br>
                Enter the module number/name for which you want to save the module display window.""")
        
        self.file_name_method = cps.Choice("Select method for constructing file names",
                                           [FN_FROM_IMAGE, FN_SEQUENTIAL,
                                            FN_SINGLE_NAME, FN_WITH_METADATA,
                                            FN_IMAGE_FILENAME_WITH_METADATA],
                                            FN_FROM_IMAGE,doc="""
                <i>(Used only if saving non-movie files)</i><br>
                Four choices are available:
                <ul>
                <li><i>From image filename:</i> The filename will be constructed based
                on the original filename of an input image specified in <b>LoadImages</b>
                or <b>LoadData</b>. You will have the opportuity to prefix or append
                additional text.</li>
                <li><i>Sequential numbers:</i> Same as above, but in addition, each filename
                will have a number appended to the end that corresponds to
                the image cycle number (starting at 1).</li>
                <li><i>Single file name:</i> A single, fixed name will be given to the
                file, with no additional text prefixed or appended. Since the filename is fixed,
                this file will be overwritten with each cycle. Unless you want the file to be 
                updated every cycle during the analysis run, you would probably want to save it
                on the last cycle (see the <a href='#when_to_save'><i>Select how often to save</i></a> setting)</li>
                <li><i>Name with metadata:</i> The filenames are constructed using the metadata
                associated with an image cycle in <b>LoadImages</b> or <b>LoadData</b>. This is 
                especially useful if you want your output given a unique label according to the
                metadata corresponding to an image group. The name of the metadata to substitute 
                is included in a special tag format embedded in your file specification. 
                %s. %s</li>
                <li><i>Image filename with metadata:</i> This is a combination of
                <i>From image filename</i> and <i>Name with metadata</i>. If you have metadata 
                associated with your images, you can append an extension to the image filename 
                using a metadata tag. </li>
                </ul>"""% (USING_METADATA_TAGS_REF,USING_METADATA_HELP_REF))
        
        self.file_image_name = cps.FileImageNameSubscriber("Select image name for file prefix",
                                                           "None",doc="""
                <i>(Used only when constructing the filename from the image filename, with or without metadata)</i><br>
                Select an image loaded using <b>LoadImages</b> or <b>LoadData</b>. The original filename will be
                used as the prefix for the output filename.""")
        
        self.single_file_name = cps.Text(SINGLE_NAME_TEXT, "OrigBlue",doc="""
                <i>(Used only when constructing the filename from the image filename, a single name or a name with metadata)</i><br>
                If you are constructing the filenames using...
                <ul>
                <li><i>Single name:</i> Enter the filename text here</li>
                <li><i>Custom with metadata:</i> If you have metadata 
                associated with your images, enter the filename text with the metadata tags. %(USING_METADATA_TAGS_REF)s.   
                For example, if the <i>plate</i>, <i>well_row</i> and <i>well_column</i> tags have the values <i>XG45</i>, <i>A</i>
                and <i>01</i>, respectively, the string <i>Illum_\g&lt;plate&gt;_\g&lt;well_row&gt;\g&lt;well_column&gt;</i>
                produces the output filename <i>Illum_XG45_A01</i>.</li>
                </ul>
                Do not enter the file extension in this setting; it will be appended automatically."""%globals())
        
        self.wants_file_name_suffix = cps.Binary(
            "Do you want to add a suffix to the image file name?", False,
            doc = """Check this setting to add a suffix to the image's file name.
            Leave the setting unchecked to use the image name as-is.""")
        
        self.file_name_suffix = cps.Text("Text to append to the image name",cps.DO_NOT_USE,doc="""
                <i>(Used only when constructing the filename from the image filename)</i><br>
                Enter the text that should be appended to the filename specified above.""")
        
        self.file_format = cps.Choice("Select file format to use",
                                      [FF_BMP,FF_GIF,FF_HDF,FF_JPG,FF_JPEG,
                                       FF_PBM,FF_PCX,FF_PGM,FF_PNG,FF_PNM,
                                       FF_PPM,FF_RAS,FF_TIF,FF_TIFF,FF_XWD,
                                       FF_MAT],FF_BMP,doc="""
                <i>(Used only when saving non-movie files)</i><br>
                Select the image or movie format to save the image(s). Most common
                image formats are available; MAT-files are readable by MATLAB.""")
        
        self.pathname = SaveImagesDirectoryPath("Output file location", doc = """ 
                <i>(Used only when saving non-movie files)</i><br>
                This setting lets you choose the folder for the output
                files. %(IO_FOLDER_CHOICE_HELP_TEXT)s
                <p>An additional option is the following:
                <ul>
                <li><i>Same folder as image</i>: Place the output file in the same folder
                that the source image is located.</li>
                </ul></p>
                <p>%(IO_WITH_METADATA_HELP_TEXT)s %(USING_METADATA_TAGS_REF)s. 
                For instance, if you have a metadata tag named 
                "Plate", you can create a per-plate folder by selecting one the subfolder options
                and then specifying the subfolder name as <i>\g&lt;Plate&gt;</i>. The module will 
                substitute the metadata values for the current image set for any metadata tags in the 
                folder name.%(USING_METADATA_HELP_REF)s.</p>"""%globals())
        
        # TODO: 
        self.bit_depth = cps.Choice("Image bit depth",
                ["8","16"],doc="""
                <i>(Used only when saving files in a non-MAT format)</i><br>
                What is the bit-depth at which you want to save the images?
                <b>16-bit images are supported only for TIF formats.
                Currently, saving images in 12-bit is not supported.</b>""")
        
        self.overwrite = cps.Binary("Overwrite existing files without warning?",False,doc="""
                Check this box to automatically overwrite a file if it already exists. Otherwise, you
                will be prompted for confirmation first.""")
        
        self.when_to_save = cps.Choice("Select how often to save",
                [WS_EVERY_CYCLE,WS_FIRST_CYCLE,WS_LAST_CYCLE],
                WS_EVERY_CYCLE,doc="""<a name='when_to_save'>
                <i>(Used only when saving non-movie files)</i><br>
                Specify at what point during pipeline execution to save file(s). </a>
                <ul>
                <li><i>Every cycle:</i> Useful for when the image of interest is created every cycle and is
                not dependent on results from a prior cycle.</li>
                <li><i>First cycle:</i> Useful for when you are saving an aggregate image created 
                on the first cycle, e.g., <b>CorrectIlluminationCalculate</b> with the <i>All</i>
                setting used on images obtained directly from <b>LoadImages</b>/<b>LoadData</b></a>.</li>
                <li><i>Last cycle:</i> Useful for when you are saving an aggregate image completed 
                on the last cycle, e.g., <b>CorrectIlluminationCalculate</b> with the <i>All</i>
                setting used on intermediate images generated during each cycle.</li>
                </ul> """)
        
        self.rescale = cps.Binary("Rescale the images? ",False,doc="""
                <i>(Used only when saving non-MAT file images)</i><br>
                Check this box if you want the image to occupy the full dynamic range of the bit 
                depth you have chosen. For example, if you save an image to an 8-bit file, the
                smallest grayscale value will be mapped to 0 and the largest value will be mapped 
                to 2<sup>8</sup>-1 = 255. 
                <p>This will increase the contrast of the output image but will also effectively 
                stretch the image data, which may not be desirable in some 
                circumstances. See <b>RescaleIntensity</b> for other rescaling options.</p>""")
        
        self.colormap = cps.Colormap('Select colormap', 
                                     value = CM_GRAY,
                                     doc= """
                <i>(Used only when saving non-MAT file images)</i><br>
                This affects how images color intensities are displayed. All available colormaps can be seen 
                <a href="http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps">here</a>.""")
        
        self.update_file_names = cps.Binary("Update file names within CellProfiler?",False,doc="""
                This setting stores filename and pathname data for each of the new files created 
                via this module, as a per-image measurement. 
                <p>This setting is useful when exporting measurements to a database, allowing 
                access to the saved image. If you are using the machine-learning tools or image
                viewer in CellProfiler Analyst, for example, you will want to check this box if you want
                the images you are saving via this module to be displayed along with the original images.</p>
                <p>This setting also allows downstream modules (e.g., <b>CreateWebPage</b>) to look up the newly
                saved files on the hard drive. Normally, whatever files are present on
                the hard drive when CellProfiler processing begins (and when the
                <b>LoadImages</b> module processes its first cycle) are the only files 
                recognized within CellProfiler. This setting allows the newly saved files
                to be accessible to downstream modules. This setting might yield unusual
                consequences if you are using the <b>SaveImages</b> module to save an image
                directly as loaded (e.g., using the <b>SaveImages</b> module to convert file
                formats), because in some places in the output file it will overwrite
                the file names of the loaded files with the file names of the saved
                files. Because this function is rarely needed and might introduce
                complications, the default setting is unchecked.""")
        
        self.create_subdirectories = cps.Binary("Create subfolders in the output folder?",False,
                                                doc = """Subfolders will be created to match the input image folder structure.""")
    
    def settings(self):
        """Return the settings in the order to use when saving"""
        return [self.save_image_or_figure, self.image_name, self.figure_name,
                self.file_name_method, self.file_image_name,
                self.single_file_name, self.wants_file_name_suffix, 
                self.file_name_suffix, self.file_format,
                self.pathname, self.bit_depth,
                self.overwrite, self.when_to_save,
                self.rescale, self.colormap, 
                self.update_file_names, self.create_subdirectories]
    
    def visible_settings(self):
        """Return only the settings that should be shown"""
        result = [self.save_image_or_figure]
        if self.save_image_or_figure == IF_FIGURE:
            result.append(self.figure_name)
        else:
            result.append(self.image_name)

        result.append(self.file_name_method)
        if (self.file_name_method not in 
            (FN_FROM_IMAGE, FN_IMAGE_FILENAME_WITH_METADATA) and
            self.pathname.dir_choice == PC_WITH_IMAGE):
            # Need just the file image name here to associate
            # the file-name image path
            result.append(self.file_image_name)
        if self.file_name_method == FN_FROM_IMAGE:
            result += [self.file_image_name, self.wants_file_name_suffix]
            if self.wants_file_name_suffix:
                result.append(self.file_name_suffix)
        elif self.file_name_method == FN_IMAGE_FILENAME_WITH_METADATA:
            self.single_file_name.text = METADATA_NAME_TEXT
            result += [self.file_image_name, self.single_file_name]
        elif self.file_name_method == FN_SEQUENTIAL:
            self.single_file_name.text = SEQUENTIAL_NUMBER_TEXT
            result.append(self.single_file_name)
        elif self.file_name_method == FN_SINGLE_NAME:
            self.single_file_name.text = SINGLE_NAME_TEXT
            result.append(self.single_file_name)
        elif self.file_name_method == FN_WITH_METADATA:
            self.single_file_name.text = METADATA_NAME_TEXT
            result.append(self.single_file_name)
        else:
            raise NotImplementedError("Unhandled file name method: %s"%(self.file_name_method))
        if self.save_image_or_figure != IF_MOVIE:
            result.append(self.file_format)
        if (self.file_format in FF_SUPPORTING_16_BIT and 
            self.save_image_or_figure == IF_IMAGE and
            has_bioformats):
            # TIFF supports 8 & 16-bit, all others are written 8-bit
            result.append(self.bit_depth)
        result.append(self.pathname)
        result.append(self.overwrite)
        if self.save_image_or_figure != IF_MOVIE:
            result.append(self.when_to_save)
        if (self.save_image_or_figure == IF_IMAGE and
            self.file_format != FF_MAT):
            result.append(self.rescale)
            result.append(self.colormap)
        if self.file_name_method in (
            FN_FROM_IMAGE, FN_SEQUENTIAL, FN_WITH_METADATA, 
            FN_IMAGE_FILENAME_WITH_METADATA):
            result.append(self.update_file_names)
            result.append(self.create_subdirectories)
        return result
    
    @property
    def module_key(self):
        return "%s_%d"%(self.module_name, self.module_num)
    
    def get_dictionary(self, image_set_list):
        '''Return the runtime dictionary associated with this module'''
        return image_set_list.legacy_fields[self.module_key]
    
    def prepare_run(self, pipeline, image_set_list, *args):
        image_set_list.legacy_fields[self.module_key] = {}
        return True

    def prepare_group(self, pipeline, image_set_list, 
                      grouping, image_numbers):
        d = self.get_dictionary(image_set_list)
        d['FIRST_IMAGE'] = True
        if self.save_image_or_figure == IF_MOVIE:
            d['N_FRAMES'] = len(image_numbers)
            d['CURRENT_FRAME'] = 0
        return True
    
    def prepare_to_create_batch(self, pipeline, image_set_list, fn_alter_path):
        self.pathname.alter_for_create_batch_files(fn_alter_path)
        
    def run(self,workspace):
        """Run the module
        
        pipeline     - instance of CellProfiler.Pipeline for this run
        workspace    - the workspace contains:
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - display within this frame (or None to not display)
        """
        if self.save_image_or_figure.value in (IF_IMAGE, IF_MASK, IF_CROPPING):
            should_save = self.run_image(workspace)
        elif self.save_image_or_figure == IF_MOVIE:
            should_save = self.run_movie(workspace)
        else:
            raise NotImplementedError(("Saving a %s is not yet supported"%
                                       (self.save_image_or_figure)))
        workspace.display_data.filename = self.get_filename(workspace)
        
    def is_interactive(self):
        # if we overwrite files, it's safe to run in the background
        return not self.overwrite.value

    def display(self, workspace):
        if workspace.frame != None:
            if self.save_image_or_figure == IF_MOVIE:
                return
            figure = workspace.create_or_find_figure(subplots=(1,1))
            outcome = ("Wrote %s" if workspace.display_data.wrote_image
                       else "Did not write %s")
            figure.subplot_table(0, 0, [[outcome %
                                         (workspace.display_data.filename)]])

    
    def run_image(self,workspace):
        """Handle saving an image"""
        #
        # First, check to see if we should save this image
        #
        if self.when_to_save == WS_FIRST_CYCLE:
            d = self.get_dictionary(workspace.image_set_list)
            if not d["FIRST_IMAGE"]:
                workspace.display_data.wrote_image = False
                self.save_filename_measurements(workspace)
                return False
            d["FIRST_IMAGE"] = False
            
        elif self.when_to_save == WS_LAST_CYCLE:
            workspace.display_data.wrote_image = False
            self.save_filename_measurements( workspace)
            return False
        self.save_image(workspace)
        return True
    
    
    def run_movie(self, workspace):
        assert has_bioformats
        d = self.get_dictionary(workspace.image_set_list)
        out_file = self.get_filename(workspace)
        if d["CURRENT_FRAME"] == 0 and os.path.exists(out_file):
            if not self.check_overwrite(out_file):
                d["CURRENT_FRAME"] = "Ignore"
                return
            else:
                # Have to delete the old movie before making the new one
                os.remove(out_file)
        elif d["CURRENT_FRAME"] == "Ignore":
            return
            
        env = jutil.attach()
        try:
            image = workspace.image_set.get_image(self.image_name.value)
            pixels = image.pixel_data
            width = pixels.shape[1]
            height = pixels.shape[0]
            if pixels.ndim == 2:
                channels = 1
            elif pixels.ndim == 3 and pixels.shape[2] == 3:
                channels = 3
            else:
                raise 'Image shape is not supported for saving to movie'
            stacks = 1
            frames = d['N_FRAMES']
            
            FormatTools = make_format_tools_class()
            imeta = createOMEXMLMetadata()
            meta = wrap_imetadata_object(imeta)
            meta.createRoot()
            meta.setPixelsBigEndian(True, 0, 0)
            meta.setPixelsDimensionOrder('XYCZT', 0, 0)
            meta.setPixelsPixelType(FormatTools.getPixelTypeString(FormatTools.UINT8), 0, 0)
            meta.setPixelsSizeX(width, 0, 0)
            meta.setPixelsSizeY(height, 0, 0)
            meta.setPixelsSizeC(channels, 0, 0)
            meta.setPixelsSizeZ(stacks, 0, 0)
            meta.setPixelsSizeT(frames, 0, 0)
            meta.setLogicalChannelSamplesPerPixel(channels, 0, 0)   
            ImageWriter = make_image_writer_class()
            writer = ImageWriter()    
            writer.setMetadataRetrieve(meta)
            writer.setId(out_file)

            is_last_image = (d['CURRENT_FRAME'] == d['N_FRAMES']-1)
            if is_last_image:
                print "Writing last image of %s" %out_file
            image = workspace.image_set.get_image(self.image_name.value)
            pixels = image.pixel_data
            pixels = (pixels*255).astype(np.uint8)
            if len(pixels.shape)==3 and pixels.shape[2] == 3:  
                save_im = np.array([pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]]).flatten()
            else:
                save_im = pixels.flatten()
            writer.saveBytes(env.make_byte_array(save_im), is_last_image)
            writer.close()
            d['CURRENT_FRAME'] += 1
        finally:
            jutil.detach()
        
    
    def post_group(self, workspace, *args):
        if (self.when_to_save == WS_LAST_CYCLE and 
            self.save_image_or_figure != IF_MOVIE):
            self.save_image(workspace)
        

    def save_image_with_bioformats(self, workspace):
        ''' Saves using bioformats library. Currently used for saving 16-bit
        tiffs. Some code is redundant from save_image, but it's easier to 
        separate the logic completely.
        '''
        assert self.file_format in (FF_TIF, FF_TIFF)
        assert self.save_image_or_figure == IF_IMAGE
        assert has_bioformats
        
        workspace.display_data.wrote_image = False

        # get the filename and check overwrite before attaching to java bridge
        filename = self.get_filename(workspace)
        path=os.path.split(filename)[0]
        if len(path) and not os.path.isdir(path):
            os.makedirs(path)
        if not self.check_overwrite(filename):
            return
        if os.path.isfile(filename):
            # Important: bioformats will append to files by default, so we must
            # delete it explicitly if it exists.
            os.remove(filename)
        
        # Get the image data to be written
        image = workspace.image_set.get_image(self.image_name.value)
        pixels = image.pixel_data
        
        if self.rescale.value:
            # Normalize intensities for each channel
            pixels = pixels.astype(np.float32)
            if pixels.ndim == 3:
                # RGB
                for i in range(3):
                    img_min = np.min(pixels[:,:,i])
                    img_max = np.max(pixels[:,:,i])
                    pixels[:,:,i] = (pixels[:,:,i] - img_min) / (img_max - img_min)
            else:
                # Grayscale
                img_min = np.min(pixels)
                img_max = np.max(pixels)
                pixels = (pixels - img_min) / (img_max - img_min)
        
        env = jutil.attach()
        try:
            width = pixels.shape[1]
            height = pixels.shape[0]
            if pixels.ndim == 2:
                channels = 1
            elif pixels.ndim == 3 and pixels.shape[2] == 3:
                channels = 3
            else:
                raise 'Image shape is not supported'
            stacks = 1
            frames = 1
            is_big_endian = (sys.byteorder.lower() == 'big')
            FormatTools = make_format_tools_class()
            pixel_type = FormatTools.getPixelTypeString(FormatTools.UINT16)
            
            # Build bioformats metadata object
            imeta = createOMEXMLMetadata()
            meta = wrap_imetadata_object(imeta)
            meta.createRoot()
            meta.setPixelsBigEndian(is_big_endian, 0, 0)
            meta.setPixelsDimensionOrder('XYCZT', 0, 0)
            meta.setPixelsPixelType(pixel_type, 0, 0)
            meta.setPixelsSizeX(width, 0, 0)
            meta.setPixelsSizeY(height, 0, 0)
            meta.setPixelsSizeC(channels, 0, 0)
            meta.setPixelsSizeZ(stacks, 0, 0)
            meta.setPixelsSizeT(frames, 0, 0)
            meta.setLogicalChannelSamplesPerPixel(channels, 0, 0)
            ImageWriter = make_image_writer_class()
            writer = ImageWriter()    
            writer.setMetadataRetrieve(meta)
            writer.setId(filename)

            if pixels.dtype in (np.uint8, np.int16):
                # Leave the values alone, but cast to unsigned int 16
                pixels = pixels.astype(np.uint16)
            elif pixels.dtype in (np.uint32, np.uint64, np.int32, np.int64):
                sys.stderr.write("Warning: converting %s image to 16-bit could result in incorrect values.\n" % repr(pixels.dtype))
                pixels = pixels.astype(np.uint16)
            elif issubclass(pixels.dtype.type, np.floating):
                # Scale pixel vals to 16 bit
                pixels = (pixels * 65535).astype(np.uint16)

            if len(pixels.shape) == 3 and pixels.shape[2] == 3:  
                pixels = np.array([pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]])

            pixels = pixels.flatten()
            # split the 16-bit image into byte-sized chunks for saveBytes
            pixels = np.fromstring(pixels.tostring(), dtype=np.uint8)
                
            writer.saveBytes(env.make_byte_array(pixels), True)
            writer.close()
        
            workspace.display_data.wrote_image = True
            
            if self.when_to_save != WS_LAST_CYCLE:
                self.save_filename_measurements(workspace)
        finally:
            jutil.detach()
                        
    def save_image(self, workspace):
        if self.get_bit_depth()== '16':
            return self.save_image_with_bioformats(workspace)
        
        workspace.display_data.wrote_image = False
        image = workspace.image_set.get_image(self.image_name.value)
        if self.save_image_or_figure == IF_IMAGE:
            pixels = image.pixel_data
            if self.file_format != FF_MAT:
                if self.rescale.value:
                    pixels = pixels.copy()
                    # Normalize intensities for each channel
                    if pixels.ndim == 3:
                        # RGB
                        for i in range(3):
                            img_min = np.min(pixels[:,:,i])
                            img_max = np.max(pixels[:,:,i])
                            pixels[:,:,i] = (pixels[:,:,i] - img_min) / (img_max - img_min)
                    else:
                        # Grayscale
                        img_min = np.min(pixels)
                        img_max = np.max(pixels)
                        pixels = (pixels - img_min) / (img_max - img_min)
                else:
                    # Clip at 0 and 1
                    if np.max(pixels) > 1 or np.min(pixels) < 0:
                        sys.stderr.write(
                            "Warning, clipping image %s before output. Some intensities are outside of range 0-1" %
                            self.image_name.value)
                        pixels = pixels.copy()
                        pixels[pixels < 0] = 0
                        pixels[pixels > 1] = 1
                        
                if pixels.ndim == 2 and self.colormap != CM_GRAY:
                    # Convert grayscale image to rgb for writing
                    if self.colormap == cps.DEFAULT:
                        colormap = cpp.get_default_colormap()
                    else:
                        colormap = self.colormap.value
                    cm = matplotlib.cm.get_cmap(colormap)
                    
                    if self.get_bit_depth() == '8':
                        pixels = mapper.to_rgba(pixels, bytes=True)
                    else:
                        raise NotImplementedError("12 and 16-bit images not yet supported")
                elif self.get_bit_depth() == '8':
                    pixels = (pixels*255).astype(np.uint8)
                else:
                    raise NotImplementedError("12 and 16-bit images not yet supported")
                
        elif self.save_image_or_figure == IF_MASK:
            pixels = image.mask.astype(np.uint8) * 255
            
        elif self.save_image_or_figure == IF_CROPPING:
            pixels = image.crop_mask.astype(np.uint8) * 255
            
        filename = self.get_filename(workspace)
        path=os.path.split(filename)[0]
        if len(path) and not os.path.isdir(path):
            os.makedirs(path)
        if pixels.ndim == 3 and pixels.shape[2] == 4:
            mode = 'RGBA'
        elif pixels.ndim == 3:
            mode = 'RGB'
        else:
            mode = 'L'
        filename = self.get_filename(workspace)

        if not self.check_overwrite(filename):
            return
        if self.get_file_format() == FF_MAT:
            scipy.io.matlab.mio.savemat(filename,{"Image":pixels},format='5')
        else:
            pil = PILImage.fromarray(pixels,mode)
            pil.save(filename, self.get_file_format())
        workspace.display_data.wrote_image = True
        if self.when_to_save != WS_LAST_CYCLE:
            self.save_filename_measurements(workspace)
        
    def check_overwrite(self, filename):
        '''Check to see if it's legal to overwrite a file
        
        Throws an exception if can't overwrite and no GUI.
        Returns False if can't overwrite otherwise
        '''
        if not self.overwrite.value and os.path.isfile(filename):
            if cpp.get_headless():
                raise 'SaveImages: trying to overwrite %s in headless mode, but Overwrite files is set to "No"'%(filename)
            else:
                import wx
                over = wx.MessageBox("Do you want to overwrite %s?"%(filename),
                                     "Warning: overwriting file", wx.YES_NO)
                if over == wx.NO:
                    return False
        return True
        
    def save_filename_measurements(self, workspace):
        if self.update_file_names.value:
            filename = self.get_filename(workspace)
            pn, fn = os.path.split(filename)
            workspace.measurements.add_measurement('Image',
                                                   self.file_name_feature,
                                                   fn)
            workspace.measurements.add_measurement('Image',
                                                   self.path_name_feature,
                                                   pn)
    
    @property
    def file_name_feature(self):
        return 'FileName_%s'%(self.image_name.value)
    
    @property
    def path_name_feature(self):
        return 'PathName_%s'%(self.image_name.value)
    
    def get_measurement_columns(self, pipeline):
        if self.update_file_names.value:
            return [(cellprofiler.measurements.IMAGE, 
                     self.file_name_feature,
                     cellprofiler.measurements.COLTYPE_VARCHAR_FILE_NAME),
                    (cellprofiler.measurements.IMAGE,
                     self.path_name_feature,
                     cellprofiler.measurements.COLTYPE_VARCHAR_PATH_NAME)]
        else:
            return []
        
    def get_filename(self,workspace):
        "Concoct a filename for the current image based on the user settings"
        
        measurements=workspace.measurements
        if self.file_name_method == FN_SINGLE_NAME:
            filename = self.single_file_name.value
        elif self.file_name_method == FN_WITH_METADATA:
            filename = self.single_file_name.value
            filename = workspace.measurements.apply_metadata(filename)
        elif self.file_name_method == FN_SEQUENTIAL:
            filename = self.single_file_name.value
            filename = '%s%d'%(filename, measurements.image_set_number+1)
        else:
            file_name_feature = 'FileName_%s'%(self.file_image_name)
            filename = measurements.get_current_measurement('Image',
                                                            file_name_feature)
            filename = os.path.splitext(filename)[0]
            if self.file_name_method == FN_IMAGE_FILENAME_WITH_METADATA:
                filename += workspace.measurements.apply_metadata(
                    self.single_file_name.value)
            elif self.wants_file_name_suffix:
                filename += str(self.file_name_suffix)
        
        filename = "%s.%s"%(filename,self.get_file_format())
        pathname = self.pathname.get_absolute_path(measurements,
                                                   self.file_image_name.value)
        
        return os.path.join(pathname,filename)
    
    def get_file_format(self):
        """Return the file format associated with the extension in self.file_format
        """
        if self.save_image_or_figure == IF_MOVIE:
            return FF_MOV
        if self.file_format == FF_JPG:
            return FF_JPEG
        if self.file_format == FF_TIF:
            return FF_TIFF
        return self.file_format.value
    
    def get_bit_depth(self):
        if (self.save_image_or_figure == IF_IMAGE and 
            self.get_file_format() in FF_SUPPORTING_16_BIT):
            return self.bit_depth.value
        else:
            return '8'
    
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        """Adjust the setting values to be backwards-compatible with old versions
        
        """
        
        PC_DEFAULT     = "Default output folder"

        if from_matlab and variable_revision_number == 12:
            # self.create_subdirectories.value is already False by default.
            variable_revision_number = 13
        if from_matlab and variable_revision_number == 13:
            new_setting_values = list(setting_values)
            for i in [3, 12]:
                if setting_values[i] == '\\':
                    new_setting_values[i] == cps.DO_NOT_USE
            variable_revision_number = 14
        if from_matlab and variable_revision_number == 14:
            new_setting_values = []
            if setting_values[0].isdigit():
                new_setting_values.extend([IF_FIGURE,setting_values[1]])
            elif setting_values[3] == 'avi':
                new_setting_values.extend([IF_MOVIE, setting_values[0]])
            elif setting_values[0].startswith("Cropping"):
                new_setting_values.extend([IF_CROPPING, 
                                           setting_values[0][len("Cropping"):]])
            elif setting_values[0].startswith("CropMask"):
                new_setting_values.extend([IF_MASK, 
                                           setting_values[0][len("CropMask"):]])
            else:
                new_setting_values.extend([IF_IMAGE, setting_values[0]])
            new_setting_values.append(new_setting_values[1])
            if setting_values[1] == 'N':
                new_setting_values.extend([FN_SEQUENTIAL,"None","None"])
            elif setting_values[1][0] == '=':
                new_setting_values.extend([FN_SINGLE_NAME,setting_values[1][1:],
                                           setting_values[1][1:]])
            else:
                if len(cellprofiler.measurements.find_metadata_tokens(setting_values[1])):
                    new_setting_values.extend([FN_WITH_METADATA, setting_values[1],
                                               setting_values[1]])
                else:
                    new_setting_values.extend([FN_FROM_IMAGE, setting_values[1],
                                               setting_values[1]])
            new_setting_values.extend(setting_values[2:4])
            if setting_values[4] == '.':
                new_setting_values.extend([PC_DEFAULT, "None"])
            elif setting_values[4] == '&':
                new_setting_values.extend([PC_WITH_IMAGE, "None"])
            else:
                if len(cellprofiler.measurements.find_metadata_tokens(setting_values[1])):
                    new_setting_values.extend([PC_WITH_METADATA,
                                               setting_values[4]])
                else:
                    new_setting_values.extend([PC_CUSTOM, setting_values[4]])
            new_setting_values.extend(setting_values[5:11])
            #
            # Last value is there just to display some text in Matlab
            #
            new_setting_values.extend(setting_values[12:-1])
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 1
            
        if not from_matlab and variable_revision_number == 1:
            # The logic of the question about overwriting was reversed.            
            if setting_values[11] == cps.YES:
                setting_values[11] = cps.NO
            else: 
                setting_values[11] = cps.YES       
            variable_revision_number = 2
            
        if (not from_matlab) and variable_revision_number == 2:
            # Default image/output directory -> Default Image Folder
            if setting_values[8].startswith("Default output"):
                setting_values = (setting_values[:8] +
                                  [PC_DEFAULT]+ setting_values[9:])
            elif setting_values[8].startswith("Same"):
                setting_values = (setting_values[:8] +
                                  [PC_WITH_IMAGE] + setting_values[9:])
            variable_revision_number = 3
            
        if (not from_matlab) and variable_revision_number == 3:
            # Changed save type from "Figure" to "Module window"
            if setting_values[0] == "Figure":
                setting_values[0] = IF_FIGURE
            setting_values = standardize_default_folder_names(setting_values,8)
            variable_revision_number = 4

        if (not from_matlab) and variable_revision_number == 4:
            save_image_or_figure, image_name, figure_name,\
	    file_name_method, file_image_name, \
	    single_file_name, file_name_suffix, file_format, \
	    pathname_choice, pathname, bit_depth, \
	    overwrite, when_to_save, \
            when_to_save_movie, rescale, colormap, \
            update_file_names, create_subdirectories = setting_values

            pathname = SaveImagesDirectoryPath.static_join_string(
                pathname_choice, pathname)
            
            setting_values = [
                save_image_or_figure, image_name, figure_name,
                file_name_method, file_image_name, single_file_name, 
                file_name_suffix != cps.DO_NOT_USE,
                file_name_suffix, file_format,
                pathname, bit_depth, overwrite, when_to_save,
                rescale, colormap, update_file_names, create_subdirectories]
            variable_revision_number = 5
        
        setting_values[9] = \
            SaveImagesDirectoryPath.upgrade_setting(setting_values[9])
        
        return setting_values, variable_revision_number, from_matlab
    
    def validate_module(self, pipeline):
        if (self.save_image_or_figure in (IF_IMAGE, IF_MASK, IF_CROPPING) and
            self.when_to_save in (WS_FIRST_CYCLE, WS_EVERY_CYCLE)):
            #
            # Make sure that the image name is available on every cycle
            #
            for setting in cps.get_name_providers(pipeline,
                                                  self.image_name):
                if not setting.provided_attributes.get(cps.AVAILABLE_ON_LAST_ATTRIBUTE):
                    return
            #
            # If we fell through, then you can only save on the last cycle
            #
            raise cps.ValidationError("%s is only available after processing all images in an image group" %
                                      self.image_name.value,
                                      self.when_to_save)
    
class SaveImagesDirectoryPath(cps.DirectoryPath):
    '''A specialized version of DirectoryPath to handle saving in the image dir'''
    
    def __init__(self, text, doc):
        super(SaveImagesDirectoryPath, self).__init__(
            text, dir_choices = [
                cps.DEFAULT_OUTPUT_FOLDER_NAME, cps.DEFAULT_INPUT_FOLDER_NAME,
                PC_WITH_IMAGE, cps.ABSOLUTE_FOLDER_NAME,
                cps.DEFAULT_OUTPUT_SUBFOLDER_NAME, 
                cps.DEFAULT_INPUT_SUBFOLDER_NAME], doc=doc)
        
    def get_absolute_path(self, measurements, image_name):
        if self.dir_choice == PC_WITH_IMAGE:
            path_name_feature = "PathName_%s" % image_name
            return measurements.get_current_image_measurement(path_name_feature)
        return super(SaveImagesDirectoryPath, self).get_absolute_path(measurements)
    
    def test_valid(self, pipeline):
        if self.dir_choice not in self.dir_choices:
            raise cps.ValidationError("%s is not a valid directory option" %
                                      self.dir_choice, self)
        
    @staticmethod
    def upgrade_setting(value):
        '''Upgrade setting from previous version'''
        dir_choice, custom_path = cps.DirectoryPath.split_string(value)
        if dir_choice in OLD_PC_WITH_IMAGE_VALUES:
            dir_choice = PC_WITH_IMAGE
        elif dir_choice in (PC_CUSTOM, PC_WITH_METADATA):
            if custom_path.startswith('.'):
                dir_choice = cps.DEFAULT_OUTPUT_SUBFOLDER_NAME
            elif custom_path.startswith('&'):
                dir_choice = cps.DEFAULT_INPUT_SUBFOLDER_NAME
                custom_path = '.' + custom_path[1:]
            else:
                dir_choice = cps.ABSOLUTE_FOLDER_NAME
        else:
            return cps.DirectoryPath.upgrade_setting(value)
        return cps.DirectoryPath.static_join_string(dir_choice, custom_path)
                  
