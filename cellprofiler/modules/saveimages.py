'''<b>Save Images </b> saves image or movie files
<hr>

Because CellProfiler usually performs many image analysis steps on many
groups of images, it does <i>not</i> save any of the resulting images to the
hard drive unless you specifically choose to do so with the <b>SaveImages</b> 
module. You can save any of the
processed images created by CellProfiler during the analysis using this module.

<p>You can choose from among 18 image formats for saving your files. This
allows you to use the module as a file format converter, by loading files
in their original format and then saving them in an alternate format.

<p>Note that saving images in 12- or 16-bit is not supported.
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
import Image as PILImage
import scipy.io.matlab.mio

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements
import cellprofiler.settings as cps
import cellprofiler.preferences as cpp
from cellprofiler.gui.help import USING_METADATA_TAGS_REF, USING_METADATA_HELP_REF
from cellprofiler.preferences import standardize_default_folder_names, DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME

IF_IMAGE       = "Image"
IF_MASK        = "Mask"
IF_CROPPING    = "Cropping"
IF_FIGURE      = "Module window"
IF_MOVIE       = "Movie"
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
PC_WITH_IMAGE  = "Same folder as image"
PC_CUSTOM      = "Custom"
PC_WITH_METADATA = "Custom with metadata"
WS_EVERY_CYCLE = "Every cycle"
WS_FIRST_CYCLE = "First cycle"
WS_LAST_CYCLE  = "Last cycle"
CM_GRAY        = "gray"

class SaveImages(cpm.CPModule):

    module_name = "SaveImages"
    variable_revision_number = 4
    category = "File Processing"
    
    def create_settings(self):
        self.save_image_or_figure = cps.Choice("Select the type of image to save",
                                               [IF_IMAGE, IF_MASK, IF_CROPPING, IF_MOVIE,IF_FIGURE],IF_IMAGE,doc="""
                The following types of images can be saved as a file on the hard drive:
                <ul>
                <li><i>Image:</i> Any of the images produced upstream of the module can be selected for saving. 
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
                <li><i>Movie:</i> A sequence of images can be saved as a movie file, such as an AVI. Each 
                image becomes a frame of the movie. <b>Currently, this option is not yet implemented.</b></li>
                <li><i>Module display window:</i> The window associated with a module can be saved, which
                will include all the panels and text within that window. <b>Currently, this option is not yet implemented.</b></li>
                </ul>
                Note that objects cannot be directly saved with the <b>SaveImages</b> module.
                You must first use the <b>ConvertObjectsToImage</b> module to convert the objects to an image, 
                followed by <b>SaveImages</b>.""")
        
        self.image_name  = cps.ImageNameSubscriber("Select the image to save","None", doc = """
                <i>(Used only if saving images, crop masks and image croppings)</i><br>
                What did you call the images you want to save?""")
        
        self.figure_name = cps.FigureSubscriber("Select the module display window to save","None",doc="""
                <i>(Used only if saving module display windows)</i><br>
                Enter the module number/name for which you want to save the module display window.""")
        
        self.file_name_method = cps.Choice("Select method for constructing file names",
                                           [FN_FROM_IMAGE,FN_SEQUENTIAL,
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
                associated with your images, enter the filename text with the metadata tags. %s.   
                For example, if the <i>plate</i>, <i>well_row</i> and <i>well_column</i> tags have the values <i>XG45</i>, <i>A</i>
                and <i>01</i>, respectively, the string <i>Illum_\g&lt;plate&gt;_\g&lt;well_row&gt;\g&lt;well_column&gt;</i>
                produces the output filename <i>Illum_XG45_A01</i>.</li>
                </ul>
                Do not enter the file extension in this setting; it will be appended automatically."""%(USING_METADATA_TAGS_REF))
        
        self.file_name_suffix = cps.Text("Text to append to the image name",cps.DO_NOT_USE,doc="""
                <i>(Used only when constructing the filename from the image filename)</i><br>
                Enter the text that will be appended to the filename specified above.""")
        
        self.file_format = cps.Choice("Select file format to use",
                                      [FF_BMP,FF_GIF,FF_HDF,FF_JPG,FF_JPEG,
                                       FF_PBM,FF_PCX,FF_PGM,FF_PNG,FF_PNM,
                                       FF_PPM,FF_RAS,FF_TIF,FF_TIFF,FF_XWD,
                                       FF_MAT],FF_BMP,doc="""
                <i>(Used only when saving non-movie files)</i><br>
                Select the image or movie format to save the image(s). Most common
                image formats are available; MAT-files are readable by MATLAB.""")
        
        self.pathname_choice = cps.Choice("Select location to save file",
                                          [DEFAULT_OUTPUT_FOLDER_NAME, PC_WITH_IMAGE,
                                           PC_CUSTOM, PC_WITH_METADATA],
                                          DEFAULT_OUTPUT_FOLDER_NAME, doc = """ 
                <i>(Used only when saving non-movie files)</i><br>
                Where do you want to store the file? There are four choices available:                
                <ul>
                <li><i>Default Output Folder:</i> The file will be stored in the default output
                folder.</li>
                <li><i>Same folder as image:</i> The file will be stored in the folder to which the
                images from this image cycle belong.</li>
                <li><i>Custom:</i> The file will be stored in a customizable folder. This folder 
                can be referenced against the default input or output folder.</li>
                <li><i>Custom with metadata:</i> Same as <i>Custon</i> but also with metadata substitution 
                (see the <i>Name with metadata</i> setting above for metadata usage)</li>
                </ul>""")
        
        self.movie_pathname_choice = cps.Choice("Select location to save file",
                                          [DEFAULT_OUTPUT_FOLDER_NAME,PC_CUSTOM],
                                          DEFAULT_OUTPUT_FOLDER_NAME,doc="""
                <i>(Used only when saving movies)</i><br>
                Where do you want to store the file? There are two choices available:                
                <ul>
                <li><i>Default Output Folder:</i> The file will be stored in the default output
                folder.</li>
                <li><i>Custom:</i> The file will be stored in a customizable folder. This folder 
                can be referenced against the default input or output folder.</li>
                </ul>""")
        
        self.pathname = cps.Text("Pathname for the saved file",".",doc="""
                <i>(Used only when using Custom or Custom with metadata to construct filenames)</i><br>
                Enter the pathname for the location where files should be saved. 
                The pathname can be an absolute path or can be referenced against the default
                folders. Pathnames that start with "." (a period) are relative to the Default Input Folder. 
                Names that start with "&" (an ampersand) are relative to the Default Output Folder. 
                Two periods ".." specify the parent folder above either of these. For example, "./MyFolder" 
                looks for a folder called <i>MyFolder</i> that is contained within the Default Input Folder,
                and "&/../MyFolder" looks in a folder called <i>MyFolder</i> at the same level as the output folder.
		<p>A useful tip: If slashes are needed to separate parts of a path, use '/' for a forward
		slash and '\\\\' for a backslash (the extra slash is to escape the backslash). Which slash you
		use will depend on your operating system.</p>""")
        
        self.bit_depth = cps.Choice("Image bit depth",
                ["8","12","16"],doc="""
                <i>(Used only when saving files in a non-MAT format)</i><br>
                What is the bit-depth that you want to save the images in?
                <b>Currently, saving images in 12- or 16-bit is not supported.</b>""")
        
        self.overwrite = cps.Binary("Overwrite existing files without warning?",False,doc="""
                Check this box to automatically overwrite a file if it already exists. Otherwise, you
                will be prompted for comfirmation first.""")
        
        self.when_to_save = cps.Choice("Select how often to save",
                [WS_EVERY_CYCLE,WS_FIRST_CYCLE,WS_LAST_CYCLE],
                WS_EVERY_CYCLE,doc="""<a name='when_to_save'>
                <i>(Used only when saving non-movie files)</i><br>
                Specify at what point during pipeline execution to save your file. </a>
                <ul>
                <li><i>Every cycle:</i> Useful for when the image is updated every cycle and is
                not dependent on results from a prior cycle.</li>
                <li><i>First cycle:</i> Useful for when you are saving an aggregate image created 
                on the first cycle, e.g., <b>CorrectIlluminationCalc</b> with the <i>All</i>
                setting used on images obtained directly from <b>LoadImages</b>/<b>LoadData</b></a>.</li>
                <li><i>Last cycle:</i> Useful for when you are saving an aggregate image completed 
                on the last cycle, e.g., <b>CorrectIlluminationCalc</b> with the <i>All</i>
                setting used on intermediate images generated during each cycle.</li>
                </ul> """)
        
        self.when_to_save_movie = cps.Choice("Select how often to save",
                                             [WS_LAST_CYCLE,"1","2","3","4","5","10","20"],
                                             WS_LAST_CYCLE,doc="""
                <i>(Used only when saving movies)</i><br>
                Specify at what point during pipeline execution to save your movie. 
                The movie will be always be saved after the last cycle is processed; since 
                a movie frame is added each cycle, saving at the last cycle will output the 
                fully completed movie. 
                <p>You also have the option to save the movie periodically 
                during image processing, so that the partial movie will be available if you cancel  
                image processing partway through. Saving movies in .avi format is 
                quite slow, so you can enter an cycle increment for saving the movie. For example, 
                entering a 1 will save the movie after every cycle. Since saving movies is 
                time-consuming, use any value other than <i>Last cycle</i> with caution.
                <p>The movie will be saved in uncompressed .avi format, which can be quite large. 
                We recommended converting the movie to a compressed movie format, 
                such as .mov using third-party software. """)
        
        self.rescale = cps.Binary("Rescale the images? ",False,doc="""
                <i>(Used only when saving images or movies)</i><br>
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
                <i>(Used only when saving images or movies)</i><br>
                This affects how images' intensities are displayed.
                The colormap choice is critical for movie (avi) files. 
                Choosing anything other than gray may degrade image 
                quality or result in image stretching.
                <p>All available colormaps can be seen 
                <a href="http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps">here</a></p>.""")
        
        self.update_file_names = cps.Binary("Update file names within CellProfiler?",False,doc="""
                This setting stores filename and pathname data for each of the new files created 
                via this module, as a per-image measurement. 
                <p>This setting is useful when exporting measurements to a database, allowing 
                access to the saved image. If you are using the machine-learning tools or image
                viewer in CellProfiler Analyst, for example, you will want to check this box if you want
                additional images to be displayed along with the original images.</p>
                <p>This setting also allows downstream modules (e.g., <b>CreateWebPage</b>) to look up the newly
                saved files on the hard drive. Normally, whatever files are present on
                the hard drive when CellProfiler processing begins (and when the
                <b>LoadImages</b> module processes its first cycle) are the only files 
                accessible within CellProfiler. This setting allows the newly saved files
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
                self.single_file_name, self.file_name_suffix, self.file_format,
                self.pathname_choice, self.pathname, self.bit_depth,
                self.overwrite, self.when_to_save,
                self.when_to_save_movie, self.rescale, self.colormap, 
                self.update_file_names, self.create_subdirectories]
    
    def visible_settings(self):
        """Return only the settings that should be shown"""
        result = [self.save_image_or_figure]
        if self.save_image_or_figure in (IF_IMAGE,IF_MASK, IF_CROPPING, IF_FIGURE):
            if self.save_image_or_figure in (IF_IMAGE, IF_MASK, IF_CROPPING):
                result.append(self.image_name)
            else:
                result.append(self.figure_name)
            result.append(self.file_name_method)
            if (self.file_name_method not in 
                (FN_FROM_IMAGE, FN_IMAGE_FILENAME_WITH_METADATA) and
                self.pathname_choice == PC_WITH_IMAGE):
                # Need just the file image name here to associate
                # the file-name image path
                result.append(self.file_image_name)
            if self.file_name_method == FN_FROM_IMAGE:
                result.append(self.file_image_name)
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
            result.append(self.file_format)
            result.append(self.pathname_choice)
            if self.pathname_choice.value in (PC_CUSTOM, PC_WITH_METADATA):
                result.append(self.pathname)
            if self.file_format != FF_MAT:
                result.append(self.bit_depth)
            result.append(self.overwrite)
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
        else:
            result.append(self.image_name)
            result.append(self.single_file_name)
            result.append(self.movie_pathname_choice)
            if self.movie_pathname_choice == PC_CUSTOM:
                result.append(self.pathname)
            result.append(self.overwrite)
            result.append(self.when_to_save_movie)
            result.append(self.rescale)
            result.append(self.colormap)
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

    def prepare_group(self, pipeline, image_set_list, *args):
        self.get_dictionary(image_set_list)["FIRST_IMAGE"] = True
        return True
    
    def prepare_to_create_batch(self, pipeline, image_set_list, fn_alter_path):
        self.pathname.value = fn_alter_path(
            self.pathname.value,
            regexp_substitution = (self.pathname_choice == PC_WITH_METADATA))
        
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
        else:
            raise NotImplementedError(("Saving a %s is not yet supported"%
                                       (self.save_image_or_figure)))
        workspace.display_data.filename = self.get_filename(workspace)
        
    def is_interactive(self):
        # if we overwrite files, it's safe to run in the background
        return self.overwrite.value

    def display(self, workspace):
        if workspace.frame != None:
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
    
    def post_group(self, workspace, *args):
        if self.when_to_save == WS_LAST_CYCLE:
            self.save_image(workspace)

    def save_image(self, workspace):
        workspace.display_data.wrote_image = False
        image = workspace.image_set.get_image(self.image_name.value)
        if self.save_image_or_figure == IF_IMAGE:
            pixels = image.pixel_data
            if self.file_format != FF_MAT:
                if self.rescale.value:
                    # Rescale the image intensity
                    if pixels.ndim == 3:
                        # get minima along each of the color axes (but not RGB)
                        for i in range(3):
                            img_min = np.min(pixels[:,:,i])
                            img_max = np.max(pixels[:,:,i])
                            pixels[:,:,i]=(pixels[:,:,i]-img_min) / (img_max-img_min)
                    else:
                        img_min = np.min(pixels)
                        img_max = np.max(pixels)
                        pixels=(pixels-img_min) / (img_max-img_min)
                if pixels.ndim == 2 and self.colormap != CM_GRAY:
                    if self.colormap == cps.DEFAULT:
                        colormap = cpp.get_default_colormap()
                    else:
                        colormap = self.colormap.value
                    cm = matplotlib.cm.get_cmap(colormap)
                    mapper = matplotlib.cm.ScalarMappable(cmap=cm)
                    if self.bit_depth == '8':
                        pixels = mapper.to_rgba(pixels,bytes=True)
                    else:
                        raise NotImplementedError("12 and 16-bit images not yet supported")
                elif self.bit_depth == '8':
                    pixels = (pixels*255).astype(np.uint8)
                else:
                    raise NotImplementedError("12 and 16-bit images not yet supported")
        elif self.save_image_or_figure == IF_MASK:
            pixels = image.mask.astype(np.uint8)*255
        elif self.save_image_or_figure == IF_CROPPING:
            pixels = image.crop_mask.astype(np.uint8)*255
            
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

        if not self.overwrite.value and os.path.isfile(filename):
            if cpp.get_headless():
                raise 'SaveImages: trying to overwrite %s in headless mode, but Overwrite files is set to "No"'%(filename)
            else:
                import wx
                over = wx.MessageBox("Do you want to overwrite %s?"%(filename),
                                     "Warning: overwriting file", wx.YES_NO)
                if over == wx.NO:
                    return
        if self.get_file_format() == FF_MAT:
            scipy.io.matlab.mio.savemat(filename,{"Image":pixels},format='5')
        else:
            pil = PILImage.fromarray(pixels,mode)
            pil.save(filename, self.get_file_format())
        workspace.display_data.wrote_image = True
        if self.when_to_save != WS_LAST_CYCLE:
            self.save_filename_measurements(workspace)
        
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
            filename = '%s%d'%(filename,measurements.image_set_number+1)
        else:
            file_name_feature = 'FileName_%s'%(self.file_image_name)
            filename = measurements.get_current_measurement('Image',
                                                            file_name_feature)
            filename = os.path.splitext(filename)[0]
            if self.file_name_method == FN_IMAGE_FILENAME_WITH_METADATA:
                filename += workspace.measurements.apply_metadata(
                    self.single_file_name.value)
            elif self.file_name_suffix != cps.DO_NOT_USE:
                filename += str(self.file_name_suffix)
        filename = "%s.%s"%(filename,self.file_format.value)
        
        if self.pathname_choice.value in (DEFAULT_OUTPUT_FOLDER_NAME, PC_CUSTOM, PC_WITH_METADATA):
            if self.pathname_choice == DEFAULT_OUTPUT_FOLDER_NAME:
                pathname = cpp.get_default_output_directory()
            else:
                pathname = str(self.pathname)
                if self.pathname_choice == PC_WITH_METADATA:
                    pathname = workspace.measurements.apply_metadata(pathname)
                pathname = cpp.get_absolute_path(pathname, cpp.ABSPATH_IMAGE)
            if (self.file_name_method in (
                FN_FROM_IMAGE, FN_SEQUENTIAL, FN_WITH_METADATA,
                FN_IMAGE_FILENAME_WITH_METADATA) and
                self.create_subdirectories.value):
                # Get the subdirectory name
                path_name_feature = 'PathName_%s'%(self.file_image_name)
                orig_pathname = measurements.get_current_measurement('Image',
                                                              path_name_feature)
                pathname = os.path.join(pathname, orig_pathname)
                
        elif self.pathname_choice == PC_WITH_IMAGE:
            path_name_feature = 'PathName_%s'%(self.file_image_name)
            pathname = measurements.get_current_measurement('Image',
                                                            path_name_feature)
            # Add the root to the pathname to recover the original name
            key = 'Pathname%s'%(self.file_image_name)
            root = workspace.image_set.legacy_fields[key]
            pathname = os.path.join(root,pathname)            
        else:
            raise NotImplementedError(("Unknown pathname mechanism: %s"%
                                       (self.pathname_choice)))
        
        return os.path.join(pathname,filename)
    def get_file_format(self):
        """Return the file format associated with the extension in self.file_format
        """
        if self.file_format == FF_JPG:
            return FF_JPEG
        if self.file_format == FF_TIF:
            return FF_TIFF
        return self.file_format.value
    
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
            new_setting_values.extend(setting_values[12:])
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
            variable_revision_number = 4

        # Standardize input/output directory name references
        setting_values = standardize_default_folder_names(setting_values,8)
        
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
    
