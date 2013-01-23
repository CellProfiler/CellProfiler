'''<b>Identify Secondary Objects</b> identifies objects (e.g., cell edges) using "seed" objects identified by
an <b>IdentifyPrimaryObjects</b> module (e.g., nuclei)
<hr>

This module identifies secondary objects (e.g., cell edges) based on two
inputs: 
<ol>
<li>A previous module's identification of primary objects (e.g.,
nuclei)</li>
<li>An image stained for the secondary objects (not required
for the <i>Distance - N</i> option).</li>
</ol>
<p>Each primary object is assumed to be completely contained 
within a secondary object (e.g., nuclei are completely contained within cells
stained for actin).

In order to identify the edges of secondary objects, this module performs two tasks: 
<ol>
<li>Finding the dividing lines between secondary objects which touch each other.</li> 
<li>Finding the dividing lines between the secondary objects and the
background of the image. This is done by thresholding the image stained
for secondary objects, except when using the <i>Distance - N</i> option.</li>
</ol>

After processing, the module display window for this module will
show panels with objects outlined in two colors:
<ul>
<li>Green: Primary objects</li>
<li>Red: Secondary objects</li>
</ul>
If you need to change the outline colors (e.g., due to color-blindness), you can 
make adjustments in <i>File > Preferences</i>.

The module window will also show another image where the identified 
objects are displayed with arbitrary colors: the colors themselves do not mean 
anything but simply help you distingush the various objects. 

<h3>Technical notes</h3>
The <i>Propagation</i> algorithm creates a set of secondary object labels using 
each primary object as a "seed", guided by the input image and limited to the 
foreground region as determined by the chosen thresholding method. &lambda; is 
a regularization parameter; see the help for the setting for more details. Propagation
of secondary object labels is by the shortest path to an adjacent primary object 
from the starting ("seeding") primary object. The seed-to-pixel distances are
calculated as the sum of absolute differences in a 3x3 (8-connected) image 
neighborhood, combined with &lambda; via sqrt(differences<sup>2</sup> + &lambda;<sup>2</sup>).
                           
<h4>Available measurements</h4>
<ul>
<li><i>Image features:</i>
<ul>
<li><i>Count:</i> The number of secondary objects identified.</li>
</ul>
</li>         
<li><i>Object features:</i>
<ul>
<li><i>Parent:</i> The identity of the primary object associated with each secondary 
object.</li>
<li><i>Location_X, Location_Y:</i> The pixel (X,Y) coordinates of the center of 
mass of the identified secondary objects.</li>
</ul>
</li>
</ul>

See also the other <b>Identify</b> modules.
'''

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org


import numpy as np
import os
import scipy.ndimage as scind
import scipy.misc as scimisc

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.workspace as cpw
import cellprofiler.settings as cps
import identify as cpmi
from identify import FI_IMAGE_SIZE
import cellprofiler.cpmath.threshold as cpthresh
import cellprofiler.cpmath.otsu
from cellprofiler.cpmath.propagate import propagate
from cellprofiler.cpmath.cpmorphology import fill_labeled_holes
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.cpmath.watershed import fast_watershed as watershed
from cellprofiler.cpmath.outline import outline

M_PROPAGATION = "Propagation"
M_WATERSHED_G = "Watershed - Gradient"
M_WATERSHED_I = "Watershed - Image"
M_DISTANCE_N = "Distance - N"
M_DISTANCE_B = "Distance - B"

class IdentifySecondaryObjects(cpmi.Identify):

    module_name = "IdentifySecondaryObjects"
    variable_revision_number = 8
    category = "Object Processing"
    
    def create_settings(self):
        self.primary_objects = cps.ObjectNameSubscriber("Select the input objects","Nuclei",doc="""
            What did you call the objects you want to use as "seeds" to identify a secondary 
            object around each one? By definition, each primary object must be associated with exactly one 
            secondary object and completely contained within it.""")
        
        self.objects_name = cps.ObjectNameProvider("Name the objects to be identified","Cells")
        
        self.method = cps.Choice("Select the method to identify the secondary objects",
                                 [M_PROPAGATION, M_WATERSHED_G, M_WATERSHED_I, 
                                  M_DISTANCE_N, M_DISTANCE_B],
                                 M_PROPAGATION, doc="""\
            <p>There are several methods available to find the dividing lines 
            between secondary objects which touch each other:
            <ul>
            <li><i>Propagation:</i> This method will find dividing lines
            between clumped objects where the image stained for secondary objects
            shows a change in staining (i.e., either a dimmer or a brighter line).
            Smoother lines work better, but unlike the Watershed method, small gaps
            are tolerated. This method is considered an improvement on the
            traditional <i>Watershed</i> method. The dividing lines between objects are
            determined by a combination of the distance to the nearest primary object
            and intensity gradients. This algorithm uses local image similarity to
            guide the location of boundaries between cells. Boundaries are
            preferentially placed where the image's local appearance changes
            perpendicularly to the boundary (TR Jones, AE Carpenter, P
            Golland (2005) <i>Voronoi-Based Segmentation of Cells on Image Manifolds</i>,
            ICCV Workshop on Computer Vision for Biomedical Image Applications, pp.
            535-543).</li>
           
            <li><i>%(M_WATERSHED_G)s:</i> This method uses the watershed algorithm
            (Vincent, Luc and Pierre Soille,
            <i>Watersheds in Digital Spaces: An Efficient Algorithm Based on Immersion
            Simulations</i>, IEEE Transactions of Pattern Analysis and Machine
            Intelligence, Vol. 13, No. 6, June 1991, pp. 583-598) to assign
            pixels to the primary objects which act as seeds for the watershed.
            In this variant, the watershed algorithm operates on the Sobel
            transformed image which computes an intensity gradient. This method
            works best when the image intensity drops off or increases rapidly
            near the boundary between cells.
            </li>
            <li><i>%(M_WATERSHED_I)s:</i> This method is similar to the above,
            but it uses the inverted intensity of the image for the watershed.
            The areas of lowest intensity will form the boundaries between
            cells. This method works best when there is a saddle of relatively
            low intensity at the cell-cell boundary.
            </li>
            <li><i>Distance:</i> In this method, the edges of the primary
            objects are expanded a specified distance to create the secondary
            objects. For example, if nuclei are labeled but there is no stain to help
            locate cell edges, the nuclei can simply be expanded in order to estimate
            the cell's location. This is often called the "doughnut" or "annulus" or
            "ring" approach for identifying the cytoplasm. 
            There are two methods that can be used:
            <ul>
            <li><i>%(M_DISTANCE_N)s</i>: In this method, the image of the secondary 
            staining is not used at all; the expanded objects are the 
            final secondary objects.</li> 
            <li><i>%(M_DISTANCE_B)s</i>: Thresholding of the secondary staining image is used to eliminate background
            regions from the secondary objects. This allows the extent of the
            secondary objects to be limited to a certain distance away from the edge
            of the primary objects without including regions of background.</li></ul></li>
            </ul>""" % globals())
        
        self.image_name = cps.ImageNameSubscriber("Select the input image",
                                                  "None",doc="""
            The selected image will be used to find the edges of the secondary objects.
            For <i>Distance - N</i> this will not affect object identification, only the final display.""")
        
        self.create_threshold_settings()
        
        self.distance_to_dilate = cps.Integer("Number of pixels by which to expand the primary objects",10,minval=1)
        
        self.regularization_factor = cps.Float("Regularization factor",0.05,minval=0,
                                               doc="""\
            <i>(Used only if Propagation method selected)</i> <br>
            The regularization factor &lambda; can be anywhere in the range 0 to infinity.
            This method takes two factors into account when deciding where to draw
            the dividing line between two touching secondary objects: the distance to
            the nearest primary object, and the intensity of the secondary object
            image. The regularization factor controls the balance between these two
            considerations: 
            <ul>
            <li>A &lambda; value of 0 means that the distance to the nearest
            primary object is ignored and the decision is made entirely on the
            intensity gradient between the two competing primary objects. </li>
            <li>Larger values of &lambda; put more and more weight on the distance between the two objects.
            This relationship is such that small changes in &lambda; will have fairly different 
            results (e.,g 0.01 vs 0.001). However, the intensity image is almost completely 
            ignored at &lambda; much greater than 1.</li>
            <li>At infinity, the result will look like %(M_DISTANCE_B)s, masked to the
            secondary staining image.</li>
            </ul>"""%globals())
        
        self.use_outlines = cps.Binary("Retain outlines of the identified secondary objects?",False)
        
        self.outlines_name = cps.OutlineNameProvider('Name the outline image',"SecondaryOutlines", doc="""\
            <i>(Used only if outlines are to be saved)</i><br>
            Once the outline image is named here, the outlines of the identified objects may be used by modules downstream,
            by selecting them from any drop-down image list.""")
        
        self.wants_discard_edge = cps.Binary(
            "Discard secondary objects touching the border of the image?",
            False,
            doc = """This option will discard objects which have an edge
            that falls on the border of the image. The objects are discarded
            with respect to downstream measurement modules, but they are retained in memory
            as "unedited objects"; this allows them to be considered in downstream modules that modify the
            segmentation.""")
        
        self.fill_holes = cps.Binary(
            "Fill holes in identified objects?", True,
            doc = """Check this box to fill any holes inside objects.""")
        
        self.wants_discard_primary = cps.Binary(
            "Discard the associated primary objects?",
            False,
            doc = """<i>(Used only if discarding secondary objects touching the image border)</i> <br>
            It might be appropriate to discard the primary object
            for any secondary object that touches the edge of the image.
            The module will create a new set of objects that mirrors your
            primary objects if you check this setting. The new objects
            will be identical to the old, except that the new objects
            will have objects removed if their associated secondary object
            touches the edge of the image.""")
            
        self.new_primary_objects_name = cps.ObjectNameProvider(
            "Name the new primary objects", "FilteredNuclei",
            doc = """<i>(Used only if associated primary objects are discarded)</i> <br>
            You can name the primary objects that remain after the discarding step.
            These objects will all have secondary objects
            that do not touch the edge of the image. Note that any primary object
            whose secondary object touches the edge will be retained in memory as an
            "unedited object"; this allows them to be considered in downstream modules that modify the
            segmentation.""")
        
        self.wants_primary_outlines = cps.Binary(
            "Retain outlines of the new primary objects?", False,
            doc = """<i>(Used only if associated primary objects are discarded)</i><br>
            Check this setting in order to save images of the outlines
            of the primary objects after filtering. You can save these images
            using the <b>SaveImages</b> module.""")
        
        self.new_primary_outlines_name = cps.OutlineNameProvider(
            "Name the new primary object outlines", "FilteredNucleiOutlines",
            doc = """<i>(Used only if associated primary objects are discarded and saving outlines of new primary objects)</i><br>
            You can name the outline image of the
            primary objects after filtering. You can refer to this image
            using this name in subsequent modules such as <b>SaveImages</b>.""")
    
    def settings(self):
        return [ self.primary_objects, self.objects_name,   
                 self.method, self.image_name, self.threshold_method, 
                 self.threshold_correction_factor, self.threshold_range,
                 self.object_fraction, self.distance_to_dilate, 
                 self.regularization_factor, self.outlines_name,
                 self.manual_threshold,  
                 self.binary_image, self.use_outlines,
                 self.two_class_otsu, self.use_weighted_variance,
                 self.assign_middle_to_foreground,
                 self.wants_discard_edge, self.wants_discard_primary,
                 self.new_primary_objects_name, self.wants_primary_outlines,
                 self.new_primary_outlines_name, self.thresholding_measurement,
                 self.fill_holes,
                 self.adaptive_window_method, self.adaptive_window_size]
    
    def visible_settings(self):
        result = [self.image_name, self.primary_objects, self.objects_name,  
                 self.method]
        if self.method != M_DISTANCE_N:
            result += self.get_threshold_visible_settings()
        if self.method in (M_DISTANCE_B,M_DISTANCE_N):
            result.append(self.distance_to_dilate)
        elif self.method == M_PROPAGATION:
            result.append(self.regularization_factor)
        result += [self.fill_holes, self.wants_discard_edge]
        if self.wants_discard_edge:
            result.append(self.wants_discard_primary)
            if self.wants_discard_primary:
                result += [self.new_primary_objects_name, 
                           self.wants_primary_outlines]
                if self.wants_primary_outlines:
                    result.append(self.new_primary_outlines_name)
        result.append(self.use_outlines)
        if self.use_outlines.value:
            result.append(self.outlines_name)
        return result
    
    def help_settings(self):
        return [ self.primary_objects, self.objects_name,   
                 self.method, self.image_name, self.threshold_method, 
                 self.two_class_otsu, self.use_weighted_variance,
                 self.assign_middle_to_foreground, self.object_fraction, 
                 self.adaptive_window_method, self.adaptive_window_size, 
                 self.manual_threshold,  
                 self.binary_image, self.thresholding_measurement,
                 self.threshold_correction_factor, self.threshold_range,
                 self.distance_to_dilate, 
                 self.regularization_factor,
                 self.fill_holes, self.wants_discard_edge, self.wants_discard_primary,
                 self.new_primary_objects_name, self.wants_primary_outlines,
                 self.new_primary_outlines_name, 
                 self.use_outlines, self.outlines_name]
    
    def upgrade_settings(self,
                         setting_values,
                         variable_revision_number,
                         module_name,
                         from_matlab):
        if from_matlab and variable_revision_number==1:
            NotImplementedError("Sorry, Matlab variable revision # 1 is not supported")
        if from_matlab and variable_revision_number==2:
            # Added test mode - default = no
            setting_values = list(setting_values)
            setting_values.append(cps.NO)
            variable_revision_number = 3
        if from_matlab and variable_revision_number==3:
            new_setting_values = list(setting_values) 
            if setting_values[4].isdigit():
                # User entered manual threshold
                new_setting_values[4] = cpthresh.TM_MANUAL
                new_setting_values.append(setting_values[4])
                new_setting_values.append(cps.DO_NOT_USE)
            elif (not setting_values[4] in 
                  (cpthresh.TM_OTSU_GLOBAL,cpthresh.TM_OTSU_ADAPTIVE,
                   cpthresh.TM_OTSU_PER_OBJECT,cpthresh.TM_MOG_GLOBAL,
                   cpthresh.TM_MOG_ADAPTIVE,cpthresh.TM_MOG_PER_OBJECT,
                   cpthresh.TM_BACKGROUND_GLOBAL,cpthresh.TM_BACKGROUND_ADAPTIVE,
                   cpthresh.TM_BACKGROUND_PER_OBJECT,cpthresh.TM_ROBUST_BACKGROUND,
                   cpthresh.TM_ROBUST_BACKGROUND_GLOBAL,
                   cpthresh.TM_ROBUST_BACKGROUND_ADAPTIVE,
                   cpthresh.TM_ROBUST_BACKGROUND_PER_OBJECT,
                   cpthresh.TM_RIDLER_CALVARD_GLOBAL,cpthresh.TM_RIDLER_CALVARD_ADAPTIVE,
                   cpthresh.TM_RIDLER_CALVARD_PER_OBJECT,cpthresh.TM_KAPUR_GLOBAL,
                   cpthresh.TM_KAPUR_ADAPTIVE,cpthresh.TM_KAPUR_PER_OBJECT)):
                # User entered an image name -  guess
                new_setting_values[4] = cpthresh.TM_BINARY_IMAGE
                new_setting_values.append('0')
                new_setting_values.append(setting_values[4])
            else:
                new_setting_values.append('0')
                new_setting_values.append(cps.DO_NOT_USE)
            if setting_values[10] == cps.DO_NOT_USE:
                new_setting_values.append(cps.NO)
            else:
                new_setting_values.append(cps.YES)
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 1
        if from_matlab:
            NotImplementedError("Don't know how to convert Matlab IdentifySecondary revision # %d"%(variable_revision_number))
        if variable_revision_number != self.variable_revision_number:
            NotImplementedError("Don't know how to handle IdentifySecondary revision # %d"%(variable_revision_number))
        if (not from_matlab) and variable_revision_number == 1:
            # Removed test mode
            # added Otsu parameters.
            setting_values = setting_values[:11]+setting_values[12:]
            setting_values += [cpmi.O_TWO_CLASS, cpmi.O_WEIGHTED_VARIANCE,
                               cpmi.O_FOREGROUND]
            variable_revision_number = 2
            
        if (not from_matlab) and variable_revision_number == 2:
            # Added discarding touching
            setting_values = setting_values + [cps.NO, cps.NO, "FilteredNuclei"]
            variable_revision_number = 3
            
        if (not from_matlab) and variable_revision_number == 3:
            # Added new primary outlines
            setting_values = setting_values + [cps.NO, "FilteredNucleiOutlines"]
            variable_revision_number = 4
            
        if (not from_matlab) and variable_revision_number == 4:
            # Added measurements to threshold methods
            setting_values = setting_values + ["None"]
            variable_revision_number = 5
            
        if (not from_matlab) and variable_revision_number == 5:
            # Change name of watershed option
            if setting_values[2] == "Watershed":
                setting_values[2] = M_WATERSHED_G
            variable_revision_number = 6
            
        if (not from_matlab) and variable_revision_number == 6:
            # Fill labeled holes added
            fill_holes = (cps.NO 
                          if setting_values[2] in (M_DISTANCE_B, M_DISTANCE_N) 
                          else cps.YES)
            setting_values = setting_values + [fill_holes]
            variable_revision_number = 7
        
        if (not from_matlab) and variable_revision_number == 7:
            # Added adaptive thresholding settings
            setting_values += [FI_IMAGE_SIZE, "10"]
            variable_revision_number = 8
            
        return setting_values, variable_revision_number, from_matlab

    def run(self, workspace):
        assert isinstance(workspace, cpw.Workspace)
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_grayscale = True)
        img = image.pixel_data
        mask = image.mask
        objects = workspace.object_set.get_objects(self.primary_objects.value)
        global_threshold = None
        if self.method == M_DISTANCE_N:
            has_threshold = False
        elif self.threshold_method == cpthresh.TM_BINARY_IMAGE:
            binary_image = workspace.image_set.get_image(self.binary_image.value,
                                                         must_be_binary = True)
            local_threshold = np.ones(img.shape) * np.max(img) + np.finfo(float).eps
            local_threshold[binary_image.pixel_data] = np.min(img) - np.finfo(float).eps
            global_threshold = cellprofiler.cpmath.otsu.otsu(img[mask],
                        self.threshold_range.min,
                        self.threshold_range.max)
            has_threshold = True
        else:
            local_threshold,global_threshold = self.get_threshold(img, mask, None, workspace)
            has_threshold = True
        
        if has_threshold:
            thresholded_image = img > local_threshold
        
        #
        # Get the following labels:
        # * all edited labels
        # * labels touching the edge, including small removed
        #
        labels_in = objects.unedited_segmented.copy()
        labels_touching_edge = np.hstack(
            (labels_in[0,:], labels_in[-1,:], labels_in[:,0], labels_in[:,-1]))
        labels_touching_edge = np.unique(labels_touching_edge)
        is_touching = np.zeros(np.max(labels_in)+1, bool)
        is_touching[labels_touching_edge] = True
        is_touching = is_touching[labels_in]
        
        labels_in[(~ is_touching) & (objects.segmented == 0)] = 0
        #
        # Stretch the input labels to match the image size. If there's no
        # label matrix, then there's no label in that area.
        #
        if tuple(labels_in.shape) != tuple(img.shape):
            tmp = np.zeros(img.shape, labels_in.dtype)
            i_max = min(img.shape[0], labels_in.shape[0])
            j_max = min(img.shape[1], labels_in.shape[1])
            tmp[:i_max, :j_max] = labels_in[:i_max, :j_max]
            labels_in = tmp
        
        if self.method in (M_DISTANCE_B, M_DISTANCE_N):
            if self.method == M_DISTANCE_N:
                distances,(i,j) = scind.distance_transform_edt(labels_in == 0, 
                                                               return_indices = True)
                labels_out = np.zeros(labels_in.shape,int)
                dilate_mask = distances <= self.distance_to_dilate.value 
                labels_out[dilate_mask] =\
                    labels_in[i[dilate_mask],j[dilate_mask]]
            else:
                labels_out, distances = propagate(img, labels_in, 
                                                  thresholded_image,
                                                  1.0)
                labels_out[distances>self.distance_to_dilate.value] = 0
                labels_out[labels_in > 0] = labels_in[labels_in>0] 
            if self.fill_holes:
                small_removed_segmented_out = fill_labeled_holes(labels_out)
            else:
                small_removed_segmented_out = labels_out
            #
            # Create the final output labels by removing labels in the
            # output matrix that are missing from the segmented image
            # 
            segmented_labels = objects.segmented
            segmented_out = self.filter_labels(small_removed_segmented_out,
                                               objects, workspace)
        elif self.method == M_PROPAGATION:
            labels_out, distance = propagate(img, labels_in, 
                                             thresholded_image,
                                             self.regularization_factor.value)
            if self.fill_holes:
                small_removed_segmented_out = fill_labeled_holes(labels_out)
            else:
                small_removed_segmented_out = labels_out.copy()
            segmented_out = self.filter_labels(small_removed_segmented_out,
                                               objects, workspace)
        elif self.method == M_WATERSHED_G:
            #
            # First, apply the sobel filter to the image (both horizontal
            # and vertical). The filter measures gradient.
            #
            sobel_image = np.abs(scind.sobel(img))
            #
            # Combine the image mask and threshold to mask the watershed
            #
            watershed_mask = np.logical_or(thresholded_image, labels_in > 0)
            watershed_mask = np.logical_and(watershed_mask, mask)
            #
            # Perform the first watershed
            #
            labels_out = watershed(sobel_image, 
                                   labels_in,
                                   np.ones((3,3),bool),
                                   mask=watershed_mask)
            if self.fill_holes:
                small_removed_segmented_out = fill_labeled_holes(labels_out)
            else:
                small_removed_segmented_out = labels_out.copy()
            segmented_out = self.filter_labels(small_removed_segmented_out,
                                               objects, workspace)
        elif self.method == M_WATERSHED_I:
            #
            # invert the image so that the maxima are filled first
            # and the cells compete over what's close to the threshold
            #
            inverted_img = 1-img
            #
            # Same as above, but perform the watershed on the original image
            #
            watershed_mask = np.logical_or(thresholded_image, labels_in > 0)
            watershed_mask = np.logical_and(watershed_mask, mask)
            #
            # Perform the watershed
            #
            labels_out = watershed(inverted_img, 
                                   labels_in,
                                   np.ones((3,3),bool),
                                   mask=watershed_mask)
            if self.fill_holes:
                small_removed_segmented_out = fill_labeled_holes(labels_out)
            else:
                small_removed_segmented_out = labels_out
            segmented_out = self.filter_labels(small_removed_segmented_out,
                                                objects, workspace)

        if self.wants_discard_edge and self.wants_discard_primary:
            #
            # Make a new primary object
            #
            lookup = scind.maximum(segmented_out,
                                   objects.segmented,
                                   range(np.max(objects.segmented)+1))
            lookup = fix(lookup)
            lookup[0] = 0
            lookup[lookup != 0] = np.arange(np.sum(lookup != 0)) + 1
            segmented_labels = lookup[objects.segmented]
            segmented_out = lookup[segmented_out]
            new_objects = cpo.Objects()
            new_objects.segmented = segmented_labels
            if objects.has_unedited_segmented:
                new_objects.unedited_segmented = objects.unedited_segmented
            if objects.has_small_removed_segmented:
                new_objects.small_removed_segmented = objects.small_removed_segmented
            new_objects.parent_image = objects.parent_image
            primary_outline = outline(segmented_labels)
            if self.wants_primary_outlines:
                out_img = cpi.Image(primary_outline.astype(bool),
                                    parent_image = image)
                workspace.image_set.add(self.new_primary_outlines_name.value, 
                                        out_img)
        else:
            primary_outline = outline(objects.segmented)
        secondary_outline = outline(segmented_out)

        #
        # Add the objects to the object set
        #
        objects_out = cpo.Objects()
        objects_out.unedited_segmented = small_removed_segmented_out
        objects_out.small_removed_segmented = small_removed_segmented_out
        objects_out.segmented = segmented_out
        objects_out.parent_image = image
        objname = self.objects_name.value
        workspace.object_set.add_objects(objects_out, objname)
        if self.use_outlines.value:
            out_img = cpi.Image(secondary_outline.astype(bool),
                                parent_image = image)
            workspace.image_set.add(self.outlines_name.value, out_img)
        object_count = np.max(segmented_out)
        #
        # Add the background measurements if made
        #
        measurements = workspace.measurements
        if has_threshold:
            if isinstance(local_threshold,np.ndarray):
                ave_threshold = np.mean(local_threshold)
            else:
                ave_threshold = local_threshold
            
            measurements.add_measurement(cpmeas.IMAGE,
                                         cpmi.FF_FINAL_THRESHOLD%(objname),
                                         np.array([ave_threshold],
                                                     dtype=float))
            measurements.add_measurement(cpmeas.IMAGE,
                                         cpmi.FF_ORIG_THRESHOLD%(objname),
                                         np.array([global_threshold],
                                                      dtype=float))
            wv = cpthresh.weighted_variance(img, mask, local_threshold)
            measurements.add_measurement(cpmeas.IMAGE,
                                         cpmi.FF_WEIGHTED_VARIANCE%(objname),
                                         np.array([wv],dtype=float))
            entropies = cpthresh.sum_of_entropies(img, mask, local_threshold)
            measurements.add_measurement(cpmeas.IMAGE,
                                         cpmi.FF_SUM_OF_ENTROPIES%(objname),
                                         np.array([entropies],dtype=float))
        cpmi.add_object_count_measurements(measurements, objname, object_count)
        cpmi.add_object_location_measurements(measurements, objname,
                                              segmented_out)
        #
        # Relate the secondary objects to the primary ones and record
        # the relationship.
        #
        children_per_parent, parents_of_children = \
            objects.relate_children(objects_out)
        measurements.add_measurement(self.primary_objects.value,
                                     cpmi.FF_CHILDREN_COUNT%objname,
                                     children_per_parent)
        measurements.add_measurement(objname,
                                     cpmi.FF_PARENT%self.primary_objects.value,
                                     parents_of_children)
        #
        # If primary objects were created, add them
        #
        if self.wants_discard_edge and self.wants_discard_primary:
            workspace.object_set.add_objects(new_objects,
                                             self.new_primary_objects_name.value)
            cpmi.add_object_count_measurements(measurements,
                                               self.new_primary_objects_name.value,
                                               np.max(new_objects.segmented))
            cpmi.add_object_location_measurements(measurements,
                                                  self.new_primary_objects_name.value,
                                                  new_objects.segmented)
            for parent_objects, parent_name, child_objects, child_name in (
                (objects, self.primary_objects.value,
                 new_objects, self.new_primary_objects_name.value),
                (new_objects, self.new_primary_objects_name.value,
                 objects_out, objname)):
                children_per_parent, parents_of_children = \
                    parent_objects.relate_children(child_objects)
                measurements.add_measurement(parent_name,
                                             cpmi.FF_CHILDREN_COUNT%child_name,
                                             children_per_parent)
                measurements.add_measurement(child_name,
                                             cpmi.FF_PARENT%parent_name,
                                             parents_of_children)
        if self.show_window:
            object_area = np.sum(segmented_out > 0)
            workspace.display_data.object_pct = \
                100 * object_area / np.product(segmented_out.shape)
            workspace.display_data.img = img
            workspace.display_data.segmented_out = segmented_out
            workspace.display_data.primary_outline = primary_outline
            workspace.display_data.secondary_outline = secondary_outline
            workspace.display_data.global_threshold = global_threshold

    def display(self, workspace, figure):
        object_pct = workspace.display_data.object_pct
        img = workspace.display_data.img
        primary_outline = workspace.display_data.primary_outline
        secondary_outline = workspace.display_data.secondary_outline
        segmented_out = workspace.display_data.segmented_out
        global_threshold = workspace.display_data.global_threshold

        figure.set_subplots((2, 2))
        title = "Input image, cycle #%d" % (workspace.measurements.image_number)
        figure.subplot_imshow_grayscale(0, 0, img, title)
        figure.subplot_imshow_labels(1, 0, segmented_out, "%s objects" % self.objects_name.value,
                                       sharexy = figure.subplot(0, 0))

        outline_img = np.dstack((img, img, img))
        cpmi.draw_outline(outline_img, secondary_outline > 0,
                          cpprefs.get_secondary_outline_color())
        figure.subplot_imshow(0, 1, outline_img, "%s outlines"%self.objects_name.value,
                                normalize=False,
                                sharexy = figure.subplot(0, 0))

        primary_img = np.dstack((img, img, img))
        cpmi.draw_outline(primary_img, primary_outline > 0,
                          cpprefs.get_primary_outline_color())
        cpmi.draw_outline(primary_img, secondary_outline > 0,
                          cpprefs.get_secondary_outline_color())
        figure.subplot_imshow(1, 1, primary_img,
                                "%s and %s outlines"%(self.primary_objects.value,self.objects_name.value),
                                normalize=False,
                                sharexy = figure.subplot(0, 0))
        if global_threshold is not None:
            figure.status_bar.SetFields(
                ["Threshold: %.3f" % global_threshold,
                 "Area covered by objects: %.1f %%" % object_pct])
        else:
            figure.status_bar.SetFields(
                ["Area covered by objects: %.1f %%" % object_pct])


    def filter_labels(self, labels_out, objects, workspace):
        """Filter labels out of the output
        
        Filter labels that are not in the segmented input labels. Optionally
        filter labels that are touching the edge.
        
        labels_out - the unfiltered output labels
        objects    - the objects thing, containing both segmented and 
                     small_removed labels
        """
        segmented_labels = objects.segmented
        max_out = np.max(labels_out)
        if max_out > 0:
            segmented_labels, m1 = cpo.size_similarly(labels_out, segmented_labels)
            segmented_labels[~m1] = 0
            lookup = scind.maximum(segmented_labels,labels_out,
                                   range(max_out+1))
            lookup = np.array(lookup, int)
            lookup[0] = 0
            segmented_labels_out = lookup[labels_out]
        else:
            segmented_labels_out = labels_out.copy()
        if self.wants_discard_edge:
            image = workspace.image_set.get_image(self.image_name.value)
            if image.has_mask:
                mask_border = (image.mask & ~ scind.binary_erosion(image.mask))
                edge_labels = segmented_labels_out[mask_border]
            else:
                edge_labels = np.hstack((segmented_labels_out[0,:],
                                         segmented_labels_out[-1,:],
                                         segmented_labels_out[:,0],
                                         segmented_labels_out[:,-1]))
            edge_labels = np.unique(edge_labels)
            #
            # Make a lookup table that translates edge labels to zero
            # but translates everything else to itself
            #
            lookup = np.arange(max_out+1)
            lookup[edge_labels] = 0
            #
            # Run the segmented labels through this to filter out edge
            # labels
            segmented_labels_out = lookup[segmented_labels_out]
                
        return segmented_labels_out
        
    def is_object_identification_module(self):
        '''IdentifySecondaryObjects makes secondary objects sets so it's a identification module'''
        return True
    
    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        columns = cpmi.get_object_measurement_columns(self.objects_name.value)
        columns += [(self.primary_objects.value,
                     cpmi.FF_CHILDREN_COUNT%self.objects_name.value,
                     cpmeas.COLTYPE_INTEGER),
                    (self.objects_name.value,
                     cpmi.FF_PARENT%self.primary_objects.value,
                     cpmeas.COLTYPE_INTEGER)]
        if self.method != M_DISTANCE_N:
            columns += cpmi.get_threshold_measurement_columns(self.objects_name.value)
        if self.wants_discard_edge and self.wants_discard_primary:
            columns += cpmi.get_object_measurement_columns(self.new_primary_objects_name.value)
            columns += [(self.new_primary_objects_name.value,
                         cpmi.FF_CHILDREN_COUNT%self.objects_name.value,
                         cpmeas.COLTYPE_INTEGER),
                        (self.objects_name.value,
                         cpmi.FF_PARENT%self.new_primary_objects_name.value,
                         cpmeas.COLTYPE_INTEGER)]
            columns += [(self.primary_objects.value,
                         cpmi.FF_CHILDREN_COUNT%self.new_primary_objects_name.value,
                         cpmeas.COLTYPE_INTEGER),
                        (self.new_primary_objects_name.value,
                         cpmi.FF_PARENT%self.primary_objects.value,
                         cpmeas.COLTYPE_INTEGER)]

        return columns
    
    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        object_dictionary = self.get_object_dictionary()
        categories = []
        if self.method != M_DISTANCE_N:
            categories += self.get_threshold_categories(pipeline, object_name)
        categories += self.get_object_categories(pipeline, object_name,
                                                 object_dictionary)
        return categories
      
    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        object_dictionary = self.get_object_dictionary()
        result = []
        if self.method != M_DISTANCE_N:
            result += self.get_threshold_measurements(pipeline, object_name,
                                                      category)
        result += self.get_object_measurements(pipeline, object_name,
                                               category, object_dictionary)
        return result
    
    def get_object_dictionary(self):
        '''Get the dictionary of parent child relationships
        
        see Identify.get_object_categories, Identify.get_object_measurements
        '''
        object_dictionary = { 
            self.objects_name.value: [self.primary_objects.value]
        }
        if self.wants_discard_edge and self.wants_discard_primary:
            object_dictionary[self.objects_name.value] += \
                             [self.new_primary_objects_name.value]
            object_dictionary[self.new_primary_objects_name.value] = \
                             [self.primary_objects.value]
        return object_dictionary
        
    def get_measurement_objects(self, pipeline, object_name, category, measurement):
        if self.method != M_DISTANCE_N:
            return self.get_threshold_measurement_objects(pipeline, object_name,
                                                          category, measurement,
                                                          self.objects_name.value)
        return []
                                                      
IdentifySecondary = IdentifySecondaryObjects
