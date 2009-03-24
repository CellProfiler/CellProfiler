"""saveimages - module to save images as bitmaps and movies

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import matplotlib
import numpy
import os
import PIL.Image
import wx

import cellprofiler.cpmodule as cpm
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
WS_EVERY_CYCLE = "Every cycle"
WS_FIRST_CYCLE = "First cycle"
WS_LAST_CYCLE  = "Last cycle"
CM_GRAY        = "gray"

class SaveImages(cpm.CPModule):
    """SHORT DESCRIPTION:
  Saves any image produced during the image analysis, in any image format.
  *************************************************************************
 
  Because CellProfiler usually performs many image analysis steps on many
  groups of images, it does *not* save any of the resulting images to the
  hard drive unless you use the SaveImages module to do so. Any of the
  processed images created by CellProfiler during the analysis can be
  saved using this module.
 
  You can choose from among 18 image formats to save your files in. This
  allows you to use the module as a file format converter, by loading files
  in their original format and then saving them in an alternate format.
 
  Please note that this module works for the cases we have tried, but it
  has not been extensively tested, particularly for how it handles color
  images, non-8 bit images, images coming from subdirectories, multiple
  incoming movie files, or filenames made by numerical increments.
 
  Settings:

  Do you want to save an image, the image's crop mask, the image's cropping,
  a movie or a figure window? 
  The Crop module creates a mask of pixels of interest in the image (the
  mask) and a cropping image (the cropping) which is the same size as the
  original image. The cropping image has the mask embedded in it. The Crop 
  module can trim its output image so that blank rows and columns in the 
  cropping image are removed in the output image and mask which is why the 
  cropping image can be different from the mask.
  In addition, this module can be used to save a figure or save images as
  frames of a movie.

  Update file names within CellProfiler:
  This setting stores file and path name data in handles.Pipeline 
  as well as a Per_image measurement.  This is useful when exporting to a
  database, allowing access to the saved image.  This also allows 
  downstream modules (e.g. CreateWebPage) to look up the newly
  saved files on the hard drive. Normally, whatever files are present on
  the hard drive when CellProfiler processing begins (and when the
  LoadImages module processes its first cycle) are the only files that are
  accessible within CellProfiler. This setting allows the newly saved files
  to be accessible to downstream modules. This setting might yield unusual
  consequences if you are using the SaveImages module to save an image
  directly as loaded (e.g. using the SaveImages module to convert file
  formats), because it will, in some places in the output file, overwrite
  the file names of the loaded files with the file names of the the saved
  files. Because this function is rarely needed and may introduce
  complications, the default answer is "No".
 
  Do you want to create the input image subdirectory structure in the  
  output directory?
  If the input images are located in subdirectories (such that you used 
  "Analyze all subfolders within the selected folder" in LoadImages), you 
  can re-create the subdirectory structure in the output directory. Note:
  This option can only be applied if you specified an original image for the
  filename prefix above, and not with "N" or "=DesiredFilename" options.
  Otherwise, all images will be saved in the output directory.
  
  Special notes for saving in movie format (avi):
  The movie will be saved after the last cycle is processed. You have the
  option to also save the movie periodically during image processing, so
  that the partial movie will be available in case image processing is
  canceled partway through. Saving movies in avi format is quite slow, so
  you can enter a number to save the movie after every Nth cycle. For
  example, entering a 1 will save the movie after every cycle. When working
  with very large movies, you may also want to save the CellProfiler output
  file every Nth cycle to save time, because the entire movie is stored in
  the output file (this may only be the case if you are working in
  diagnostic mode, see Set Preferences). See the SpeedUpCellProfiler
  module. If you are processing multiple movies, especially movies in
  subdirectories, you should save after every cycle (and also, be aware
  that this module has not been thoroughly tested under those conditions).
  Note also that the movie data is stored in the handles.Pipeline.Movie
  structure of the output file, so you can retrieve the movie data there in
  case image processing is aborted. At the time this module was written,
  MATLAB was only capable of saving in uncompressed avi format (at least on
  the UNIX platform), which is time and space-consuming. You should convert
  the results to a compressed movie format, like .mov using third-party
  software. For suggested third-party software, see the help for the
  LoadImages module.
 
  See also LoadImages, SpeedUpCellProfiler.
"""

    variable_revision_number = 1
    category = "File Processing"
    
    def create_settings(self):
        self.module_name = "SaveImages"
        self.save_image_or_figure = cps.Choice("Do you want to save an image, the image's crop mask, the image's cropping, a movie or a figure window? (see help)",
                                               [IF_IMAGE, IF_MASK, IF_CROPPING, IF_MOVIE,IF_FIGURE],IF_IMAGE)
        self.image_name  = cps.ImageNameSubscriber("What did you call the images you want to save?","None")
        self.figure_name = cps.FigureSubscriber("What figure do you want to save?","None")
        self.file_name_method = cps.Choice("How do you want to construct file names?",
                                           [FN_FROM_IMAGE,FN_SEQUENTIAL,FN_SINGLE_NAME],
                                           FN_FROM_IMAGE)
        self.file_image_name = cps.FileImageNameSubscriber("What images do you want to use for the file prefix?",
                                                           "None")
        self.single_file_name = cps.Text("What is the prefix you want to use for the single file name?",
                                         "OrigBlue")
        self.file_name_suffix = cps.Text("Enter text to append to the image name:",cps.DO_NOT_USE)
        self.file_format = cps.Choice("What file format do you want to use to save images?",
                                      [FF_BMP,FF_GIF,FF_HDF,FF_JPG,FF_JPEG,
                                       FF_PBM,FF_PCX,FF_PGM,FF_PNG,FF_PNM,
                                       FF_PPM,FF_RAS,FF_TIF,FF_TIFF,FF_XWD,
                                       FF_AVI,FF_MAT],FF_BMP)
        self.pathname_choice = cps.Choice("Where do you want to store the file?",
                                          [PC_DEFAULT,PC_WITH_IMAGE,PC_CUSTOM],
                                          PC_DEFAULT)
        self.movie_pathname_choice = cps.Choice("Where do you want to store the file?",
                                          [PC_DEFAULT,PC_CUSTOM],
                                          PC_DEFAULT)
        self.pathname = cps.Text("Enter the pathname of the directory where you want to save images:",".")
        self.bit_depth = cps.Choice("Enter the bit depth at which to save the images:",
                                    ["8","12","16"])
        self.overwrite_check = cps.Binary("Do you want to check for file overwrites?",True)
        self.when_to_save = cps.Choice("At what point do you want to save the image?",
                                       [WS_EVERY_CYCLE,WS_FIRST_CYCLE,WS_LAST_CYCLE],
                                       WS_EVERY_CYCLE)
        self.when_to_save_movie = cps.Choice("Do you want to save the movie only after the last cycle or after every Nth (1,2,3) cycle? Saving movies is time-consuming",
                                             [WS_LAST_CYCLE,"1","2","3","4","5","10","20"],
                                             WS_LAST_CYCLE)
        self.rescale = cps.Binary("Do you want to rescale the images to use the full dynamic range? Use the RescaleIntensity module for other rescaling options.",False)
        self.colormap = cps.Choice("For grayscale images, specify the colormap to use (see help). This is critical for movie (avi) files. Choosing anything other than gray may degrade image quality or result in image stretching",
                                   [CM_GRAY,"autumn","bone","cool","copper",
                                    "flag","hot","hsv","jet","pink","prism",
                                    "spring","summer","winter"],CM_GRAY)
        self.update_file_names = cps.Binary("Update file names within CellProfiler? See help for details.",False)
        self.create_subdirectories = cps.Binary("Do you want to create subdirectories in the output directory to match the input image directory structure?",False)
    
    def settings(self):
        """Return the settings in the order to use when saving"""
        return [self.save_image_or_figure, self.image_name, self.figure_name,
                self.file_name_method, self.file_image_name,
                self.single_file_name, self.file_name_suffix, self.file_format,
                self.pathname_choice, self.pathname, self.bit_depth,
                self.overwrite_check, self.when_to_save,
                self.when_to_save_movie, self.rescale, self.colormap, 
                self.update_file_names, self.create_subdirectories]
    
    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        """Adjust the setting values to be backwards-compatible with old versions
        
        """
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
                new_setting_values.extend([FN_FROM_IMAGE, setting_values[1],
                                           setting_values[1]])
            new_setting_values.extend(setting_values[2:4])
            if setting_values[4] == '.':
                new_setting_values.extend([PC_DEFAULT, "None"])
            elif setting_values[4] == '&':
                new_setting_values.extend([PC_WITH_IMAGE, "None"])
            else:
                new_setting_values.extend([PC_CUSTOM, setting_values[4]])
            new_setting_values.extend(setting_values[5:11])
            new_setting_values.extend(setting_values[12:])
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 1
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
            if self.file_name_method in (FN_FROM_IMAGE,FN_SEQUENTIAL):
                result.append(self.file_image_name)
                result.append(self.file_name_suffix)
            elif self.file_name_method == FN_SINGLE_NAME:
                result.append(self.single_file_name)
            else:
                raise NotImplementedError("Unhandled file name method: %s"%(self.file_name_method))
            result.append(self.file_format)
            result.append(self.pathname_choice)
            if self.pathname_choice == PC_CUSTOM:
                result.append(self.pathname)
            result.append(self.bit_depth)
            result.append(self.when_to_save)
            if self.save_image_or_figure == IF_IMAGE:
                result.append(self.rescale)
                result.append(self.colormap)
            if self.file_name_method in (FN_FROM_IMAGE,FN_SEQUENTIAL):
                result.append(self.update_file_names)
                result.append(self.create_subdirectories)
        else:
            result.append(self.image_name)
            result.append(self.single_file_name)
            result.append(self.movie_pathname_choice)
            if self.movie_pathname_choice == PC_CUSTOM:
                result.append(self.pathname)
            result.append(self.overwrite_check)
            result.append(self.when_to_save_movie)
            result.append(self.rescale)
            result.append(self.colormap)
        return result

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
                                       (self.save_image_or_figure.lower())))
    
    def run_image(self,workspace):
        """Handle saving an image"""
        #
        # First, check to see if we should save this image
        #
        if self.when_to_save == WS_FIRST_CYCLE:
            if workspace.measurements.image_set_number > 0:
                return
        elif self.when_to_save == WS_LAST_CYCLE:
            if (workspace.measurements.image_set_number <
                workspace.image_set_list.count()-1):
                return
            
        image = workspace.image_set.get_image(self.image_name)
        if self.save_image_or_figure == IF_IMAGE:
            pixels = image.pixel_data
            if self.rescale.value:
                # Rescale the image intensity
                if pixels.ndim == 3:
                    # get minima along each of the color axes (but not RGB)
                    for i in range(3):
                        img_min = numpy.min(pixels[:,:,i])
                        img_max = numpy.max(pixels[:,:,i])
                        pixels[:,:,i]=(pixels[:,:,i]-img_min) / (img_max-img_min)
                else:
                    img_min = numpy.min(pixels)
                    img_max = numpy.max(pixels)
                    pixels=(pixels-img_min) / (img_max-img_min)
            if pixels.ndim == 2 and self.colormap != CM_GRAY:
                cm = matplotlib.cm.get_cmap(self.colormap)
                mapper = matplotlib.cm.ScalarMappable(cmap=cm)
                if self.bit_depth == '8':
                    pixels = mapper.to_rgba(pixels,bytes=True)
                else:
                    raise NotImplementedError("12 and 16-bit images not yet supported")
            elif self.bit_depth == '8':
                pixels = (pixels*255).astype(numpy.uint8)
            else:
                raise NotImplementedError("12 and 16-bit images not yet supported")
        elif self.save_image_or_figure == IF_MASK:
            pixels = image.mask.astype(int)*255
        elif self.save_image_or_figure == IF_CROPPING:
            pixels = image.crop_mask.astype(int)*255
            
        filename = self.get_filename(workspace)
        if pixels.ndim == 3 and pixels.shape[2] == 4:
            mode = 'RGBA'
        elif pixels.ndim == 3:
            mode = 'RGB'
        else:
            mode = 'L'
        pil = PIL.Image.fromarray(pixels,mode)
        filename = self.get_filename(workspace)
        if self.overwrite_check.value and os.path.isfile(filename):
            over = wx.MessageBox("Do you want to overwrite %s?"%(filename),
                                 "Warning: overwriting file", wx.YES_NO)
            if over == wx.ID_NO:
                return
        pil.save(filename,self.get_file_format())
        if self.update_file_names.value:
            pn, fn = os.path.split(filename)
            workspace.measurements.add_measurement('Image',
                                                   'FileName_%s'%(self.image_name.value),
                                                   fn)
            workspace.measurements.add_measurement('Image',
                                                   'PathName_%s'%(self.image_name.value),
                                                   pn)
    
    def get_filename(self,workspace):
        "Concoct a filename for the current image based on the user settings"
        
        measurements=workspace.measurements
        if self.file_name_method == FN_SINGLE_NAME:
            filename = self.single_file_name.value
        else:
            file_name_feature = 'FileName_%s'%(self.file_image_name)
            filename = measurements.get_current_measurement('Image',
                                                            file_name_feature)
            filename = os.path.splitext(filename)[0]
            if self.file_name_method == FN_SEQUENTIAL:
                filename = '%s%d'%(filename,measurements.image_set_number+1)
            if self.file_name_suffix != cps.DO_NOT_USE:
                filename += str(self.file_name_suffix)
        filename = "%s.%s"%(filename,self.file_format.value)
        
        if self.pathname_choice in (PC_DEFAULT, PC_CUSTOM):
            if self.pathname_choice == PC_DEFAULT:
                pathname = cpp.get_default_output_directory()
            else:
                pathname = str(self.pathname)
                if pathname[:2]=='.'+os.path.sep:
                    pathname = os.path.join(cpp.get_default_output_directory(),
                                            pathname[2:])
            if (self.file_name_method in (FN_FROM_IMAGE,FN_SEQUENTIAL) and
                self.create_subdirectories.value):
                # Get the subdirectory name
                path_name_feature = 'PathName_%s'%(self.file_image_name)
                orig_pathname = measurements.get_current_measurement('Image',
                                                              path_name_feature)
                # Get the part of the path that's different from the root
                key = 'Pathname%s'%(self.file_image_name)
                root = workspace.image_set.legacy_fields[key]
                if orig_pathname[:len(root)] != root:
                    raise ValueError("File pathname (%s) did not match root(%s)"%(orig_pathname,root))
                pathname = os.path.join(pathname,orig_pathname[len(root)+1:])
                
        elif self.pathname_choice == PC_WITH_IMAGE:
            path_name_feature = 'PathName_%s'%(self.file_image_name)
            pathname = measurements.get_current_measurement('Image',
                                                            path_name_feature)
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
