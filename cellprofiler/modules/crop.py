'''<b>Crop</b> crops or masks an image
<hr>

This module crops images into a rectangle, ellipse, an arbitrary shape provided by
you, the shape of object(s) identified by an <b>Identify</b> module, or a shape created using a previous <b>Crop</b> module in the pipeline.

<p>Keep in mind that cropping changes the size of your images, which may
have unexpected consequences. For example, identifying objects in a
cropped image and then trying to measure their intensity in the
<i>original</i> image will not work because the two images are not the same
size.</p>

<h4>Available measurements</h4>
<ul>
<li><i>AreaRetainedAfterCropping:</i> The area of the image left after cropping.</li>
<li><i>OriginalImageArea:</i> The area of the original input image.</li>
</ul>

<i>Special note on saving images:</i> You can save the cropping shape that you have defined in this module (e.g., an ellipse
you drew) so that you can use the <i>Image</i> option in future analyses. To do
this, save either the mask or cropping in <b>SaveImages</b>. See the <b>SaveImages</b> module help for more information on saving cropping shapes.
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

import math
import numpy as np
import sys

import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.preferences as cpprefs

SH_RECTANGLE = "Rectangle"
SH_ELLIPSE = "Ellipse"
SH_IMAGE = "Image"
SH_OBJECTS = "Objects"
SH_CROPPING = "Previous cropping"
CM_COORDINATES = "Coordinates"
CM_MOUSE = "Mouse"
IO_INDIVIDUALLY = "Every"
IO_FIRST = "First"
RM_NO = "No"
RM_EDGES = "Edges"
RM_ALL = "All"

FF_AREA_RETAINED = 'Crop_AreaRetainedAfterCropping_%s'
FF_ORIGINAL_AREA = 'Crop_OriginalImageArea_%s'

OFF_IMAGE_NAME              = 0
OFF_CROPPED_IMAGE_NAME      = 1
OFF_SHAPE                   = 2
OFF_CROP_METHOD             = 3
OFF_INDIVIDUAL_OR_ONCE      = 4
OFF_HORIZONTAL_LIMITS       = 5
OFF_VERTICAL_LIMITS         = 6
OFF_CENTER                  = 7
OFF_X_RADIUS                = 8
OFF_Y_RADIUS                = 9
OFF_PLATE_FIX               = 10
OFF_REMOVE_ROWS_AND_COLUMNS = 11
OFF_IMAGE_MASK_SOURCE       = 12
OFF_CROPPING_MASK_SOURCE    = 13

D_FIRST_IMAGE_SET = "FirstImageSet"
D_FIRST_CROPPING = "FirstCropping"
D_FIRST_CROPPING_MASK = "FirstCroppingMask"

class Crop(cpm.CPModule):

    module_name = "Crop"
    variable_revision_number = 2
    category = "Image Processing"
    
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber("Select the input image","None",doc = """
                            What did you call the image to be cropped?""")
        
        self.cropped_image_name = cps.CroppingNameProvider("Name the output image","CropBlue",doc = """
                            What do you want to call the cropped image?""")
        
        self.shape=cps.Choice("Select the cropping shape",
                            [SH_RECTANGLE, SH_ELLIPSE, SH_IMAGE,
                             SH_OBJECTS, SH_CROPPING],
                            SH_ELLIPSE,doc = """
                            Into which shape would you like to crop? 
                            <ul>
                            <li><i>Rectangle:</i> Self-explanatory</li>
                            <li><i>Ellipse:</i> Self-explanatory</li>
                            <li><i>Image:</i> Cropping will occur based on a binary image you specify. A choice box with available images will appear from which 
                            you can select an image. To crop into an arbitrary shape that you define, choose 
                            <i>Image</i> and use the <b>LoadSingleImage</b> module to load a black and white image 
                            that you have already prepared from a file. If you have created this image in a 
                            program such as Photoshop, this binary image should contain only the values 0 and 255,
                            with zeros (black) for the parts you want to remove and 255 (white) for
                            the parts you want to retain. Alternately, you may have previously generated a
                            binary image using this module (e.g., using the <i>Ellipse</i> option) and saved
                            it using the <b>SaveImages</b> module.<br>
                            In any case, the image must be exactly the same starting size as your image
                            and should contain a contiguous block of white pixels, because 
                            the cropping module may remove rows and columns that are
                            completely blank.</li>
                            <li><i>Objects:</i> Crop based on labeled objects identified by a previous
                            <b>Identify</b> module.</li>
                            <li><i>Previous cropping:</i> The cropping generated by a previous cropping module.
                            A choice box with available images appears if you choose <i>Cropping</i>. The
                            images in this box are ones that were generated by previous <b>Crop</b> modules.
                            This <b>Crop</b> module will use the same cropping that was used to generate whichever image
                            you choose.</li>
                            </ul>""")
        
        self.crop_method = cps.Choice("Select the cropping method",
                            [CM_COORDINATES, CM_MOUSE], CM_COORDINATES, doc = """                                      
                            Would you like to crop by typing in pixel coordinates or clicking with the mouse?
                            For <i>Ellipse</i>, you will be asked to click five or more
                            points to define an ellipse around the part of the image you want to
                            analyze.  Keep in mind that the more points you click, the longer it will
                            take to calculate the ellipse shape. For <i>Rectangle</i>, you can click as many
                            points as you like that are in the interior of the region you wish to
                            retain.""")
        
        self.individual_or_once = cps.Choice("Apply which cycle's cropping pattern?",
                            [IO_INDIVIDUALLY, IO_FIRST],
                            IO_INDIVIDUALLY, doc = """
                            Should the cropping pattern in the first image cycle be 
                            applied to all subsequent image cycles (<i>First</i>) or 
                            should every image cycle be cropped individually (<i>Every</i>)?""")
        
        self.horizontal_limits = cps.IntegerOrUnboundedRange("Left and right rectangle positions",
                            minval=0, doc = """
                            <i>(Used if Rectangle selected as cropping shape, or if using Plate Fix)</i><br>
                            Specify the left and right positions for the bounding rectangle by selecting one of the following:<br>
                            <ul><li><i>Absolute</i> to specify these values as absolute pixel
                            coordinates in the original image. For instance, you might enter
                            "25", "225", and "Absolute" to create a 200x200 pixel image that is
                            25 pixels from the top-left corner.</li>
                            <li><i>From edge</i> to specify position relative to the original image's
                            edge. For instance, you might enter "25", "25", and "Edge" to
                            crop 25 pixels from both the left and right edges of the image, irrespective
                            of the image's original size.</li></ul>""")
        
        self.vertical_limits = cps.IntegerOrUnboundedRange("Top and bottom rectangle positions",
                            minval=0, doc = """
                            <i>(Used if Rectangle selected as cropping shape, or if using Plate Fix)</i><br>
                            Specify the top and bottom positions for the bounding rectangle by selecting one of the following:<br>
                            <ul><li><i>Absolute</i> to specify these values as absolute pixel coordinates.
                            For instance, you might enter "25", "225", and "Absolute" 
                            to create a 200x200 pixel image that's 25 pixels 
                            from the top-left corner.</li>
                            <li><i>From edge</i> to specify position relative to the image edge.
                            For instance, you might enter "25", "25", and "Edge" to
                            crop 25 pixels from the edges of your images irrespective
                            of their size.</li></ul>""")
        
        self.ellipse_center = cps.Coordinates("Coordinates of ellipse center",(500,500),doc = """
                            <i>(Used if Ellipse selected as cropping shape)</i><br>
                            What is the center pixel position of the ellipse?""")
        
        self.ellipse_x_radius = cps.Integer("Ellipse radius, X direction",400, doc = """
                            <i>(Used if Ellipse selected as cropping shape)</i><br>
                            What is the radius of the ellipse in the X direction?""")
        
        self.ellipse_y_radius = cps.Integer("Ellipse radius, Y direction",200, doc = """
                            <i>(Used if Ellipse selected as cropping shape)</i><br>
                            What is the radius of the ellipse in the Y direction?""")
        
        self.image_mask_source = cps.ImageNameSubscriber("Select the masking image","None",doc = """
                            <i>(Used if Image selected as cropping shape)</i><br>
                            What is the name of the image to use as a cropping mask?""")
        
        self.cropping_mask_source = cps.CroppingNameSubscriber("Select the image with a cropping mask","None", doc = """
                            <i>(Used if Cropping selected as cropping shape)</i><br>
                            What is the name of the image with the associated cropping mask?""")
        
        self.objects_source = cps.ObjectNameSubscriber("Select the objects","None", doc="""
                            <i>(Used if Objects selected as cropping shape)</i><br>
                            What are the objects to be used as a cropping mask?""")
        
        self.use_plate_fix = cps.Binary("Use Plate Fix?",False,doc = """
                            <i>(Used if Image selected as cropping shape)</i><br>
                            Do you want to use Plate Fix? When attempting to crop based on a previously identified object
                            such as a rectangular plate, the plate may not have
                            precisely straight edges: there might be a tiny, almost unnoticeable
                            "appendage" sticking out. Without Plate Fix, the <b>Crop</b>
                            module would not crop the image tightly enough: it would retain the tiny appendage, leaving a lot
                            of blank space around the plate and potentially causing problems with later
                            modules (especially IlluminationCorrection). Plate Fix takes the
                            identified object and crops to exclude any minor appendages (technically,
                            any horizontal or vertical line where the object covers less than 50% of
                            the image). It also sets pixels around the edge of the object (for
                            regions greater than 50% but less than 100%) that otherwise would be 0 to the
                            background pixel value of your image, thus avoiding problems with
                            other modules. <i>Important note:</i> Plate Fix uses the coordinates
                            entered in the boxes normally used for rectangle cropping (Top, Left and
                            Bottom, Right) to tighten the edges around your identified plate. This
                            is done because in the majority of plate identifications you do not want
                            to include the sides of the plate. If you would like the entire plate to
                            be shown, you should enter "1:end" for both coordinates. If, for example, you would like
                            to crop 80 pixels from each edge of the plate, you could enter Top, Left and Bottom, Right values of 80 and select <i>From edge</i>.""")
        
        self.remove_rows_and_columns = cps.Choice("Remove empty rows and columns?",
                            [RM_NO, RM_EDGES, RM_ALL],
                            RM_NO, doc = """
                            Do you want to remove rows and columns that lack objects? Options are:
                            <ul>
                            <li><i>No:</i> Leave the image the same size. The cropped areas will be trned to black (zeroes)</li>
                            <li><i>Edges:</i> Crop the image so that its top, bottom, left and right are at
                            the first nonblank pixel for that edge</li>
                            <li><i>All:</i> Remove any row or column of all-blank pixels, even from the
                            internal portion of the image</li>
                            </ul>""")
    
    def settings(self):
        return [self.image_name, self.cropped_image_name, self.shape,
                self.crop_method, self.individual_or_once,
                self.horizontal_limits, self.vertical_limits,
                self.ellipse_center, self.ellipse_x_radius, 
                self.ellipse_y_radius, self.use_plate_fix,
                self.remove_rows_and_columns, self.image_mask_source,
                self.cropping_mask_source, self.objects_source]
    
    def visible_settings(self):
        result = [self.image_name, self.cropped_image_name, self.shape]
        if self.shape.value in ( SH_RECTANGLE, SH_ELLIPSE):
            result += [self.crop_method, self.individual_or_once]
            if (self.crop_method == CM_COORDINATES):
                if self.shape == SH_RECTANGLE:
                    result += [self.horizontal_limits, self.vertical_limits]
                elif self.shape == SH_ELLIPSE:
                    result += [self.ellipse_center, self.ellipse_x_radius,
                               self.ellipse_y_radius]
        elif self.shape == SH_IMAGE:
            result += [self.image_mask_source, self.use_plate_fix]
            if self.use_plate_fix.value:
                result += [self.horizontal_limits, self.vertical_limits]
        elif self.shape == SH_CROPPING:
            result.append(self.cropping_mask_source)
        elif self.shape == SH_OBJECTS:
            result.append(self.objects_source)
        else:
            raise NotImplementedError("Unimplemented shape type: %s"%(self.shape.value))
        result += [self.remove_rows_and_columns]
        return result
   
    def prepare_group(self, pipeline, image_set_list, grouping,
                      image_numbers):
        '''Prepare to start processing a new grouping
        
        pipeline - the pipeline being run
        image_set_list - the image_set_list for the experiment. Add image
                         providers to the image set list here.
        grouping - a dictionary that describes the key for the grouping.
                   For instance, { 'Metadata_Row':'A','Metadata_Column':'01'}
        image_numbers - a sequence of the image numbers within the
                   group (image sets can be retreved as
                   image_set_list.get_image_set(image_numbers[i]-1)
        
        prepare_group is called once after prepare_run if there are no
        groups.
        '''
        d = self.get_dictionary(image_set_list)
        d[D_FIRST_IMAGE_SET] = True
    
    def run(self,workspace):
        d = self.get_dictionary(workspace.image_set_list)
        first_image_set = d[D_FIRST_IMAGE_SET]
        d[D_FIRST_IMAGE_SET] = False
        orig_image = workspace.image_set.get_image(self.image_name.value)
        recalculate_flag = (self.shape not in (SH_ELLIPSE, SH_RECTANGLE) or
                            self.individual_or_once == IO_INDIVIDUALLY or
                            first_image_set)
        save_flag = (self.individual_or_once == IO_FIRST and first_image_set)
        if not recalculate_flag:
            if d[D_FIRST_CROPPING].shape != orig_image.pixel_data.shape[:2]:
                recalculate_flag = True
                sys.stderr.write("""Image, "%s", size changed from %s to %s during cycle %d, recalculating"""%
                                 (self.image_name.value, 
                                  str(self.__first_cropping.shape),
                                  str(orig_image.pixel_data.shape[:2]),
                                  workspace.image_set.number+1))
        mask = None # calculate the mask after cropping unless set below
        cropping = None
        masking_objects = None
        if not recalculate_flag:
            cropping = d[D_FIRST_CROPPING]
            mask = d[D_FIRST_CROPPING_MASK]
        elif self.shape == SH_CROPPING:
            cropping_image = workspace.image_set.get_image(self.cropping_mask_source.value)
            cropping = cropping_image.crop_mask
        elif self.shape == SH_IMAGE:
            source_image = workspace.image_set.get_image\
                (self.image_mask_source.value).pixel_data
            if self.use_plate_fix.value:
                source_image = self.plate_fixup(source_image)
            cropping = source_image > 0
        elif self.shape == SH_OBJECTS:
            masking_objects = workspace.get_objects(self.objects_source.value)
            cropping = masking_objects.segmented > 0
        elif self.crop_method == CM_MOUSE:
            cropping = self.ui_crop(workspace,orig_image)
        elif self.shape == SH_ELLIPSE:
            cropping = self.get_ellipse_cropping(workspace,orig_image)
        elif self.shape == SH_RECTANGLE:
            cropping = self.get_rectangle_cropping(workspace,orig_image)
        if self.remove_rows_and_columns == RM_NO:
            cropped_pixel_data = orig_image.pixel_data.copy()
            if cropped_pixel_data.ndim == 3:
                cropped_pixel_data[~cropping,:]=0
            else:
                cropped_pixel_data[np.logical_not(cropping)] = 0
            if mask == None:
                mask = cropping
        else:
            internal_cropping = self.remove_rows_and_columns == RM_ALL
            cropped_pixel_data = cpi.crop_image(orig_image.pixel_data,
                                                cropping,
                                                internal_cropping)
            if mask == None:
                mask = cpi.crop_image(cropping, cropping, internal_cropping)
            if cropped_pixel_data.ndim == 3:
                cropped_pixel_data[~mask,:] = 0
            else:
                cropped_pixel_data[~mask] = 0
        if self.shape == SH_OBJECTS:
            # Special handling for objects - masked objects instead of
            # mask and crop mask
            output_image = cpi.Image(image=cropped_pixel_data,
                                     masking_objects = masking_objects,
                                     parent_image = orig_image)
        else:
            output_image=cpi.Image(image=cropped_pixel_data,
                                   mask=mask,
                                   parent_image = orig_image,
                                   crop_mask = cropping)
        #
        # Display the image
        #
        if workspace.frame != None:
            window_name = "CellProfiler(%s:%d)"%(self.module_name,self.module_num)
            my_frame=workspace.create_or_find_figure(
                        title="Crop image #%d"%(self.module_num), 
                        window_name=window_name, subplots=(2,1))
            
            title = "Original: %s, cycle # %d"%(self.image_name.value,
                                      workspace.image_set.number+1)
            my_frame.subplot_imshow_grayscale(0,0,orig_image.pixel_data,title)
            my_frame.subplot_imshow_bw(1,0,cropped_pixel_data,
                                       self.cropped_image_name.value)
        if save_flag:
            d[D_FIRST_CROPPING_MASK] = mask
            d[D_FIRST_CROPPING] = cropping
        #
        # Save the image / cropping / mask
        #
        workspace.image_set.add(self.cropped_image_name.value, output_image)
        #
        # Save the old and new image sizes
        #
        original_image_area = np.product(orig_image.pixel_data.shape[:2])
        area_retained_after_cropping = np.sum(cropping) 
        feature = FF_AREA_RETAINED%(self.cropped_image_name.value)
        m = workspace.measurements
        m.add_measurement('Image', feature,
                          np.array([area_retained_after_cropping]))
        feature = FF_ORIGINAL_AREA%(self.cropped_image_name.value)
        m.add_measurement('Image', feature,
                          np.array([original_image_area]))
    
    def get_measurement_columns(self, pipeline):
        '''Return information on the measurements made during cropping'''
        return [(cpmeas.IMAGE,
                 x % self.cropped_image_name.value,
                 cpmeas.COLTYPE_INTEGER)
                for x in (FF_AREA_RETAINED, FF_ORIGINAL_AREA)]                                        
    
    def ui_crop(self, workspace,orig_image):
        """Crop into a rectangle or ellipse, guided by UI"""
        raise NotImplementedError("Cropping using the mouse has not been implemented")
    
    def get_ellipse_cropping(self, workspace,orig_image):
        """Crop into an ellipse using user-specified coordinates"""
        pixel_data = orig_image.pixel_data
        x_max = pixel_data.shape[1]
        y_max = pixel_data.shape[0]
        x_center = self.ellipse_center.x
        y_center = self.ellipse_center.y
        x_radius = self.ellipse_x_radius.value
        y_radius = self.ellipse_y_radius.value
        if x_radius > y_radius:
            dist_x = math.sqrt(x_radius**2-y_radius**2)
            dist_y = 0
            major_radius = x_radius
        else:
            dist_x = 0
            dist_y = math.sqrt(y_radius**2-x_radius**2)
            major_radius = y_radius
        
        focus_1_x,focus_1_y = (x_center-dist_x,y_center-dist_y)
        focus_2_x,focus_2_y = (x_center+dist_x,y_center+dist_y)
        y,x = np.mgrid[0:y_max,0:x_max]
        d1 = np.sqrt((x-focus_1_x)**2+(y-focus_1_y)**2)
        d2 = np.sqrt((x-focus_2_x)**2+(y-focus_2_y)**2)
        cropping = d1+d2 <= major_radius*2
        return cropping
    
    def get_rectangle_cropping(self, workspace,orig_image):
        """Crop into a rectangle using user-specified coordinates"""
        cropping = np.ones(orig_image.pixel_data.shape[:2],bool)
        if not self.horizontal_limits.unbounded_min:
            cropping[:,:self.horizontal_limits.min]=False
        if not self.horizontal_limits.unbounded_max:
            cropping[:,self.horizontal_limits.max:]=False
        if not self.vertical_limits.unbounded_min:
            cropping[:self.vertical_limits.min,:]=False
        if not self.vertical_limits.unbounded_max:
            cropping[self.vertical_limits.max:,:]=False
        return cropping
        
    def plate_fixup(self,pixel_data):
        """Fix up the cropping image based on the plate fixup rules
        
        The rules:
        * Trim rows and columns off of the edges if less than 50%
        * Use the horizontal and vertical trim to trim the image further
        """ 
        pixel_data = pixel_data.copy()
        i_histogram = pixel_data.sum(axis=1)
        i_cumsum    = np.cumsum(i_histogram > pixel_data.shape[0]/2)
        j_histogram = pixel_data.sum(axis=0)
        j_cumsum    = np.cumsum(j_histogram > pixel_data.shape[1]/2)
        i_first     = np.argwhere(i_cumsum==1)[0]
        i_last      = np.argwhere(i_cumsum==i_cumsum.max())[0]
        i_end       = i_last+1
        j_first     = np.argwhere(j_cumsum==1)[0]
        j_last      = np.argwhere(j_cumsum==j_cumsum.max())[0]
        j_end       = j_last+1
        if not self.horizontal_limits.unbounded_min:
            j_first = max(j_first,self.horizontal_limits.min)
        if not self.horizontal_limits.unbounded_max:
            j_end = min(j_end, self.horizontal_limits.max)
        if not self.vertical_limits.unbounded_min:
            i_first = max(i_first,self.vertical_limits.min)
        if not self.vertical_limits.unbounded_max:
            i_end = min(i_end, self.vertical_limits.max)
        if i_first > 0:
            if pixel_data.ndim == 3:
                pixel_data[:i_first,:,:] = 0
            else:
                pixel_data[:i_first,:] = 0
        if i_end < pixel_data.shape[0]:
            if pixel_data.ndim == 3:
                pixel_data[i_end:,:,:] = 0                
            else:
                pixel_data[i_end:,:] = 0
        if j_first > 0:
            if pixel_data.ndim == 3:
                pixel_data[:,:j_first,:] = 0
            else:
                pixel_data[:,:j_first] = 0
        if j_end < pixel_data.shape[1]:
            if pixel_data.ndim == 3:
                pixel_data[:,j_end:,:] = 0
            else:
                pixel_data[:,j_end:] = 0
        return pixel_data
        
    def upgrade_settings(self,setting_values,variable_revision_number,
                         module_name,from_matlab):
        if from_matlab and variable_revision_number==4:
            # Added OFF_REMOVE_ROWS_AND_COLUMNS
            new_setting_values = list(setting_values)
            new_setting_values.append(cps.NO)
            variable_revision_number = 5
        if from_matlab and variable_revision_number==5:
            # added image mask source, cropping mask source and reworked
            # the shape to add SH_IMAGE and SH_CROPPING
            new_setting_values = list(setting_values)
            new_setting_values.extend(["None","None","None"])
            shape = setting_values[OFF_SHAPE]
            if shape not in (SH_ELLIPSE, SH_RECTANGLE):
                # the "shape" is the name of some image file. If it
                # starts with Cropping, then it's the crop mask of
                # some other image
                if shape.startswith('Cropping'):
                    new_setting_values[OFF_CROPPING_MASK_SOURCE] =\
                        shape[len('Cropping'):]
                    new_setting_values[OFF_SHAPE] = SH_CROPPING
                else:
                    new_setting_values[OFF_IMAGE_MASK_SOURCE] = shape
                    new_setting_values[OFF_SHAPE] = SH_IMAGE
            if new_setting_values[OFF_REMOVE_ROWS_AND_COLUMNS] == cps.YES:
                new_setting_values[OFF_REMOVE_ROWS_AND_COLUMNS] = RM_EDGES
            setting_values = new_setting_values
            variable_revision_number = 2
            from_matlab = False
        
        if (not from_matlab) and variable_revision_number == 1:
            # Added ability to crop objects
            new_setting_values = list(setting_values)
            new_setting_values.append("None")
            variable_revision_number = 2
        
        if variable_revision_number == 2 and not from_matlab:
            # minor - "Cropping" changed to "Previous cropping"
            setting_values = list(setting_values)
            if setting_values[OFF_SHAPE] == "Cropping":
                setting_values[OFF_SHAPE] = SH_CROPPING
            #
            # Individually changed to "every"
            #
            if setting_values[OFF_INDIVIDUAL_OR_ONCE] == "Individually":
                setting_values[OFF_INDIVIDUAL_OR_ONCE] = IO_INDIVIDUALLY
        return setting_values, variable_revision_number, from_matlab
    
