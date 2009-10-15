'''<b>Save Images </b> saves any image produced during the image analysis, in any image format
<hr>
Because CellProfiler usually performs many image analysis steps on many
groups of images, it does <i>not</i> save any of the resulting images to the
hard drive unless you use the <b>SaveImages</b> module to do so. Any of the
processed images created by CellProfiler during the analysis can be
saved using this module.

<p>You can choose from among 18 image formats to save your files in. This
allows you to use the module as a file format converter, by loading files
in their original format and then saving them in an alternate format.

<p>Please note that this module works for the cases we have tried, but it
has not been extensively tested. In particular, saving of images in 12- or 16-bit
is not supported.
 
See also <b>LoadImages</b>, <b>SpeedUpCellProfiler</b>.
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

import matplotlib
import numpy as np
import os
import Image as PILImage
import scipy.io.matlab.mio
import wx

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements
import cellprofiler.settings as cps
import cellprofiler.preferences as cpp

IF_IMAGE       = "Image"
IF_MASK        = "Mask"
IF_CROPPING    = "Cropping"
IF_FIGURE      = "Figure"
IF_MOVIE       = "Movie"
FN_FROM_IMAGE  = "From image filename"
FN_SEQUENTIAL  = "Sequential numbers"
FN_SINGLE_NAME = "Single name"
SINGLE_NAME_TEXT = "What is the single file name?"
FN_WITH_METADATA = "Name with metadata"
METADATA_NAME_TEXT = ("""What is the file name with metadata?""")
SEQUENTIAL_NUMBER_TEXT = "What is the file prefix for sequentially numbered files?"
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
PC_DEFAULT     = "Default output directory"
PC_WITH_IMAGE  = "Same directory as image"
PC_CUSTOM      = "Custom"
PC_WITH_METADATA = "Custom with metadata"
WS_EVERY_CYCLE = "Every cycle"
WS_FIRST_CYCLE = "First cycle"
WS_LAST_CYCLE  = "Last cycle"
CM_GRAY        = "gray"

class SaveImages(cpm.CPModule):

    variable_revision_number = 2
    category = "File Processing"
    
    def create_settings(self):
        self.module_name = "SaveImages"
        self.save_image_or_figure = cps.Choice("Do you want to save an image, the image's crop mask, the image's cropping, a movie or a figure window?",
                                               [IF_IMAGE, IF_MASK, IF_CROPPING, IF_MOVIE,IF_FIGURE],IF_IMAGE,doc="""
                <ul>
                <li><i>Image:</i> Any the images produced upstream of the module can be selected for saving.</li>
                <li><i>Crop mask (Only relevant if the Crop module is used):</i> The <b>Crop</b> module 
                creates a mask of the pixels of interest in the image. Saving the mask will produce a 
                binary image in which the pixels of interest are set to 1; otherwise the pixels are 
                set to 0 elsewhere.</li>
                <li><i>Image's cropping (Only relevant if the Crop module is used):</i> The <b>Crop</b> 
                module also creates a cropping image which is typically the same size as the original 
                image. However, since the <b>Crop</b> permits removal the rows and columns that are left 
                blank, the cropping can be of a different size than the mask.</li>
                <li><i>Movie:</i> A sequence of images can be saved as a movie file, such as an AVI. Each image 
                is a frame of the movie.</li>
                <li><i>Figure window:</i> The window associated with a module can be saved, which
                will include all the panels and text within that window.</li>
                </ul>""")
        self.image_name  = cps.ImageNameSubscriber("What did you call the images you want to save?","None")
        self.figure_name = cps.FigureSubscriber("What figure do you want to save?","None",doc="""
                Select the module number/name for which you want to save the figure window.""")
        self.file_name_method = cps.Choice("How do you want to construct file names?",
                                           [FN_FROM_IMAGE,FN_SEQUENTIAL,
                                            FN_SINGLE_NAME, FN_WITH_METADATA],
                                           FN_FROM_IMAGE,doc="""
                There are four choices available:
                <ul>
                <li><i>From image filename:</i> The filename will be constructed based
                on the original filename of an input image specified in <b>LoadImages</b>
                or <b>LoadText</b>.</li>
                <li><i>Sequential numbers:</i> Same as above, but in addition, each filename
                will have a number appended to the end, corresponding to
                the image set cycle number (starting at 1).</li>
                <li><i>Single file name:</i> A single, fixed name will be given to the
                file. Since this file will therefore be overwritten with each cycyle, usually 
                this option is used if want to have the file updated on every cycle as processing 
                continues.</li>
                <li><i>Name with metadata:</i> The filenames are constructed using the metadata
                associated with an image set in <b>LoadImages</b> or <b>LoadText</b>. This is 
                especially useful if you want your output given a unique label according to the
                metadata corresponding to an image group. The name of the metadata to substitute 
                is included in a special tag format embedded in your file specification. 
                Tags have the form <i>\g&lt;metadata-tag&gt;</i> where
                <i>&lt;metadata-tag&gt</i>; is the name of your tag.</li>
                </ul>""")
        self.file_image_name = cps.FileImageNameSubscriber("What images do you want to use for the file prefix?",
                                                           "None",doc="""
                Select an image loaded using <b>LoadImages</b> or <b>LoadText</b>. The orginal filename will be
                used as the prefix for the output filename.""")
        self.single_file_name = cps.Text(SINGLE_NAME_TEXT, "OrigBlue",doc="""
                If you are constructing the filenames using:<br>
                <ul>
                <li><i>Single name:</i> Enter the filename text here</li>
                <li><i>Custom with metadata:</i> Enter the filename text with the metdata tags in the form <i>\g&lt;metadata-tag&gt</i>.  
                For example, if the <i>plate</i>, <i>well_row</i> and <i>well_column</i> tags have the values <i>XG45</i>, <i>A</i>
                and <i>01</i>, respectively, the string <i>Illum_\g&lt;plate&gt;_\g&lt;well_row&gt;\g&lt;well_column&gt;</i>
                produces the output filename <i>Illum_XG45_A01</i>.</li>
                </ul>""")
        self.file_name_suffix = cps.Text("Enter text to append to the image name:",cps.DO_NOT_USE,doc="""
                Enter the text that will be appended to the filename specified above.""")
        self.file_format = cps.Choice("What file format do you want to use to save images?",
                                      [FF_BMP,FF_GIF,FF_HDF,FF_JPG,FF_JPEG,
                                       FF_PBM,FF_PCX,FF_PGM,FF_PNG,FF_PNM,
                                       FF_PPM,FF_RAS,FF_TIF,FF_TIFF,FF_XWD,
                                       FF_AVI,FF_MAT],FF_BMP,doc="""
                Select the image or movie format to save the image(s). Most of the 
                most-common formats are supported.""")
        self.pathname_choice = cps.Choice("Where do you want to store the file?",
                                          [PC_DEFAULT,PC_WITH_IMAGE,
                                           PC_CUSTOM, PC_WITH_METADATA],
                                          PC_DEFAULT)
        self.movie_pathname_choice = cps.Choice("Where do you want to store the file?",
                                          [PC_DEFAULT,PC_CUSTOM],
                                          PC_DEFAULT,doc="""
                This setting lets you control the directory used to store the file. The
                choices are:
                <ul>
                <li><i>Default output directory</i></li>
                <li><i>Same directory as image:</i> The file will be stored in the directory of the
                images from this image set.</li>
                <li><i>Custom:</i> The file will be stored in a customizable directory. You can
                prefix the directory name with "." (an period) to make the root directory the default
                output directory or "&" (an ampersand) to make the root directory the default image
                directory.</li>
                <li><i>Custom with metadata:</i> The file will be stored in a customizable directory
                with metadata substitution (see the <i>Name with metadata</i> setting above)</li>
                </ul>""")
        self.pathname = cps.Text("Enter the pathname of the directory where you want to save images:",".",doc="""
                Enter the pathname to save the images here. The pathname can referenced with respect 
                to the Default Output directory directory with a period (".") or the Default Input 
                directory with an ampersand ("&") as the root directory.""")
        self.bit_depth = cps.Choice("Enter the bit depth at which to save the images:",
                                    ["8","12","16"])
        self.overwrite = cps.Binary("Do you want to overwrite existing files without warning?",False)
        self.when_to_save = cps.Choice("When do you want to save the image?",
                                       [WS_EVERY_CYCLE,WS_FIRST_CYCLE,WS_LAST_CYCLE],
                                       WS_EVERY_CYCLE)
        self.when_to_save_movie = cps.Choice("Do you want to save the movie only after the last cycle or after every Nth (1,2,3) cycle? ",
                                             [WS_LAST_CYCLE,"1","2","3","4","5","10","20"],
                                             WS_LAST_CYCLE,doc="""
                Specify at what point during pipeline execution to save your movie. 
                The movie will be always be saved after the last cycle is processed; since 
                a movie frame is added each cycle, saving at the last cycle will output the 
                fully completed movie. 
                <p>You also have the option to save the movie periodically 
                during image processing, so that the partial movie will be available in case 
                image processing is canceled partway through. Saving movies in .avi format is 
                quite slow, so you can enter an cycle increment to save the movie. For example, 
                entering a 1 will save the movie after every cycle. Since saving movies is 
                time-consuming, use any value other than "Last cycyle" with caution.
                <p>The movie will be saved in uncompressed avi format, which can be quite large. 
                We recommended converting the movie to a compressed movie format, 
                such as .mov using third-party software. """)
        self.rescale = cps.Binary("Do you want to rescale the images to use the full dynamic range? ",False,doc="""
                Check this box if you want the image to occupy the full dynamic range of the bit 
                depth you have chosen. For example, for a image to be saved to a 8-bit file, the
                smallest grayscale value will be mapped to 0 and the largest value will be mapped 
                to 2<sup>8</sup>-1 = 255. 
                <p>This will increase the contrast of the output image but will also effectively 
                stretch the image data, which may not be desirable in some 
                circumstances. See <b>RescaleIntensity</b> for other rescaling options.""")
        self.colormap = cps.Choice("For grayscale images, specify the colormap to use (see help). This is critical for movie (avi) files. Choosing anything other than gray may degrade image quality or result in image stretching",
                                   [CM_GRAY,"autumn","bone","cool","copper",
                                    "flag","hot","hsv","jet","pink","prism",
                                    "spring","summer","winter"],CM_GRAY)
        self.update_file_names = cps.Binary("Update file names within CellProfiler?",False,doc="""
                This setting stores file- and pathname data as a per-image measurement. 
                This is useful when exporting to a database, allowing access to the saved image.  
                This also allows downstream modules (e.g. <b>CreateWebPage</b>) to look up the newly
                saved files on the hard drive. Normally, whatever files are present on
                the hard drive when CellProfiler processing begins (and when the
                <b>LoadImages</b> module processes its first cycle) are the only files that are
                accessible within CellProfiler. This setting allows the newly saved files
                to be accessible to downstream modules. This setting might yield unusual
                consequences if you are using the <b>SaveImages</b> module to save an image
                directly as loaded (e.g. using the <b>SaveImages</b> module to convert file
                formats), because it will, in some places in the output file, overwrite
                the file names of the loaded files with the file names of the the saved
                files. Because this function is rarely needed and may introduce
                complications, the default setting is unchecked.""")
        self.create_subdirectories = cps.Binary("Do you want to create subdirectories in the output directory to match the input image directory structure?",False)
    
    def settings(self):
        """Return the settings in the order to use when saving"""
        return [self.save_image_or_figure, self.image_name, self.figure_name,
                self.file_name_method, self.file_image_name,
                self.single_file_name, self.file_name_suffix, self.file_format,
                self.pathname_choice, self.pathname, self.bit_depth,
                self.overwrite, self.when_to_save,
                self.when_to_save_movie, self.rescale, self.colormap, 
                self.update_file_names, self.create_subdirectories]
    
    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        """Adjust the setting values to be backwards-compatible with old versions
        
        """
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

        return setting_values, variable_revision_number, from_matlab
    
    def visible_settings(self):
        """Return only the settings that should be shown"""
        result = [self.save_image_or_figure]
        if self.save_image_or_figure in (IF_IMAGE,IF_MASK, IF_CROPPING, IF_FIGURE):
            if self.save_image_or_figure in (IF_IMAGE, IF_MASK, IF_CROPPING):
                result.append(self.image_name)
            else:
                result.append(self.figure_name)
            result.append(self.file_name_method)
            if (self.file_name_method != FN_FROM_IMAGE and
                self.pathname_choice == PC_WITH_IMAGE):
                # Need just the file image name here to associate
                # the file-name image path
                result.append(self.file_image_name)
            if self.file_name_method == FN_FROM_IMAGE:
                result.append(self.file_image_name)
                result.append(self.file_name_suffix)
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
            if self.file_name_method in (FN_FROM_IMAGE,FN_SEQUENTIAL,FN_WITH_METADATA):
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
            self.run_image(workspace)
        else:
            raise NotImplementedError(("Saving a %s is not yet supported"%
                                       (self.save_image_or_figure)))
    
    def run_image(self,workspace):
        """Handle saving an image"""
        #
        # First, check to see if we should save this image
        #
        if self.when_to_save == WS_FIRST_CYCLE:
            d = self.get_dictionary(workspace.image_set_list)
            if not d["FIRST_IMAGE"]:
                return
            d["FIRST_IMAGE"] = False
            
        elif self.when_to_save == WS_LAST_CYCLE:
            return
        self.save_image(workspace)
    
    def post_group(self, workspace, *args):
        if self.when_to_save == WS_LAST_CYCLE:
            self.save_image(workspace)

    def save_image(self, workspace):
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
                    cm = matplotlib.cm.get_cmap(self.colormap)
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
        if self.get_file_format() == FF_MAT:
            scipy.io.matlab.mio.savemat(filename,{"Image":pixels},format='5')
        else:
            pil = PILImage.fromarray(pixels,mode)
            if not self.overwrite.value and os.path.isfile(filename):
                over = wx.MessageBox("Do you want to overwrite %s?"%(filename),
                                     "Warning: overwriting file", wx.YES_NO)
                if over == wx.ID_NO:
                    return
            pil.save(filename,self.get_file_format())
        if self.update_file_names.value:
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
            return [(cellprofiler.measurements.IMAGE, x,
                     cellprofiler.measurements.COLTYPE_VARCHAR_FILE_NAME)
                    for x in (self.file_name_feature, self.path_name_feature)]
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
            if self.file_name_suffix != cps.DO_NOT_USE:
                filename += str(self.file_name_suffix)
        filename = "%s.%s"%(filename,self.file_format.value)
        
        if self.pathname_choice.value in (PC_DEFAULT, PC_CUSTOM, PC_WITH_METADATA):
            if self.pathname_choice == PC_DEFAULT:
                pathname = cpp.get_default_output_directory()
            else:
                pathname = str(self.pathname)
                if self.pathname_choice == PC_WITH_METADATA:
                    pathname = workspace.measurements.apply_metadata(pathname)
                if pathname[:2]=='.'+os.path.sep:
                    pathname = os.path.join(cpp.get_default_output_directory(),
                                            pathname[2:])
            if (self.file_name_method in (FN_FROM_IMAGE,FN_SEQUENTIAL,FN_WITH_METADATA) and
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
