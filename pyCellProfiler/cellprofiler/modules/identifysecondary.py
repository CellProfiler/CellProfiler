"""identifysecondary.py - Identify secondary objects surrounding primary ones

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import matplotlib
import matplotlib.cm
import numpy as np
import os
import scipy.ndimage as scind
import scipy.misc as scimisc

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.gui.cpfigure as cpf
import identify as cpmi
import cellprofiler.cpmath.threshold as cpthresh
from cellprofiler.cpmath.propagate import propagate
from cellprofiler.cpmath.cpmorphology import fill_labeled_holes
from cellprofiler.cpmath.watershed import fast_watershed as watershed
from cellprofiler.cpmath.outline import outline

M_PROPAGATION = "Propagation"
M_WATERSHED_G = "Watershed"
M_WATERSHED_I = "Watershed - Image"
M_DISTANCE_N = "Distance - N"
M_DISTANCE_B = "Distance - B"

class IdentifySecondary(cpmi.Identify):
    """% SHORT DESCRIPTION:
Identifies objects (e.g. cell edges) using "seed" objects identified by
an Identify Primary module (e.g. nuclei).
*************************************************************************

This module identifies secondary objects (e.g. cell edges) based on two
inputs: (1) a previous module's identification of primary objects (e.g.
nuclei) and (2) an image stained for the secondary objects (not required
for the Distance - N option). Each primary object is assumed to be completely
within a secondary object (e.g. nuclei are completely within cells
stained for actin).

It accomplishes two tasks:
(a) finding the dividing lines between secondary objects which touch each
other. Three methods are available: Propagation, Watershed (an older
version of Propagation), and Distance.
(b) finding the dividing lines between the secondary objects and the
background of the image. This is done by thresholding the image stained
for secondary objects, except when using Distance - N.

Settings:

Methods to identify secondary objects:
* Propagation - For task (a), this method will find dividing lines
between clumped objects where the image stained for secondary objects
shows a change in staining (i.e. either a dimmer or a brighter line).
Smoother lines work better, but unlike the watershed method, small gaps
are tolerated. This method is considered an improvement on the
traditional watershed method. The dividing lines between objects are
determined by a combination of the distance to the nearest primary object
and intensity gradients. This algorithm uses local image similarity to
guide the location of boundaries between cells. Boundaries are
preferentially placed where the image's local appearance changes
perpendicularly to the boundary. Reference: TR Jones, AE Carpenter, P
Golland (2005) Voronoi-Based Segmentation of Cells on Image Manifolds,
ICCV Workshop on Computer Vision for Biomedical Image Applications, pp.
535-543. For task (b), thresholding is used.

* Watershed - For task (a), this method will find dividing lines between
objects by looking for dim lines between objects. For task (b),
thresholding is used. Reference: Vincent, Luc, and Pierre Soille,
"Watersheds in Digital Spaces: An Efficient Algorithm Based on Immersion
Simulations," IEEE Transactions of Pattern Analysis and Machine
Intelligence, Vol. 13, No. 6, June 1991, pp. 583-598.
The watershed is done on the Sobel transform of the image which highlights
the points where the gradient is greatest. The resulting image has an annulus
around each cell which creates the watershed border between cells. The threshold
is used to mask areas outside of the secondary objects.

* Watershed - Image - For task (a), this method finds dividing lines between
objects at the saddle-points of the image. For task (b), masking using the
threshold is used.

* Distance - This method is bit unusual because the edges of the primary
objects are expanded a specified distance to create the secondary
objects. For example, if nuclei are labeled but there is no stain to help
locate cell edges, the nuclei can simply be expanded in order to estimate
the cell's location. This is often called the 'doughnut' or 'annulus' or
'ring' approach for identifying the cytoplasmic compartment. Using the
Distance - N method, the image of the secondary staining is not used at
all, and these expanded objects are the final secondary objects. Using
the Distance - B method, thresholding is used to eliminate background
regions from the secondary objects. This allows the extent of the
secondary objects to be limited to a certain distance away from the edge
of the primary objects.

Select automatic thresholding method or enter an absolute threshold:
   The threshold affects the stringency of the lines between the objects
and the background. See the help for the IdentifyPrimAutomatic module for
a complete description of the options. Note that Per object options are
not available for IdentifySecondary because the Per object method relies
on identifying objects *smaller* than the primary objects, whereas
secondary objects are always *larger* than their corresponding primary
objects.

Threshold correction factor:
When the threshold is calculated automatically, it may consistently be
too stringent or too lenient. You may need to enter an adjustment factor
which you empirically determine is suitable for your images. The number 1
means no adjustment, 0 to 1 makes the threshold more lenient and greater
than 1 (e.g. 1.3) makes the threshold more stringent. For example, the
Otsu automatic thresholding inherently assumes that 50% of the image is
covered by objects. If a larger percentage of the image is covered, the
Otsu method will give a slightly biased threshold that may have to be
corrected using a threshold correction factor.

Lower and upper bounds on threshold:
Can be used as a safety precaution when the threshold is calculated
automatically. For example, if there are no objects in the field of view,
the automatic threshold will be unreasonably low. In such cases, the
lower bound you enter here will override the automatic threshold.

Approximate percentage of image covered by objects:
An estimate of how much of the image is covered with objects. This
information is currently only used in the MoG (Mixture of Gaussian)
thresholding but may be used for other thresholding methods in the future
(see below).

Regularization factor (for propagation method only):
This method takes two factors into account when deciding where to draw
the dividing line between two touching secondary objects: the distance to
the nearest primary object, and the intensity of the secondary object
image. The regularization factor controls the balance between these two
considerations: A value of zero means that the distance to the nearest
primary object is ignored and the decision is made entirely on the
intensity gradient between the two competing primary objects. Larger
values weight the distance between the two values more and more heavily.
The regularization factor can be infinitely large, but around 10 or so,
the intensity image is almost completely ignored and the dividing line
will simply be halfway between the two competing primary objects.

Note: Primary identify modules produce two (hidden) output images that
are used by this module. The Segmented image contains the final, edited
primary objects (i.e. objects at the border and those that are too small
or large have been excluded). The SmallRemovedSegmented image is the
same except that the objects at the border and the large objects have
been included. These extra objects are used to perform the identification
of secondary object outlines, since they are probably real objects (even
if we don't want to measure them). Small objects are not used at this
stage because they are more likely to be artifactual, and so they
therefore should not "claim" any secondary object pixels.

TECHNICAL DESCRIPTION OF THE PROPAGATION OPTION:
Propagate labels from LABELS_IN to LABELS_OUT, steered by IMAGE and
limited to MASK. MASK should be a logical array. LAMBDA is a
regularization parameter, larger being closer to Euclidean distance in
the image plane, and zero being entirely controlled by IMAGE. Propagation
of labels is by shortest path to a nonzero label in LABELS_IN. Distance
is the sum of absolute differences in the image in a 3x3 neighborhood,
combined with LAMBDA via sqrt(differences^2 + LAMBDA^2). Note that there
is no separation between adjacent areas with different labels (as there
would be using, e.g., watershed). Such boundaries must be added in a
postprocess.

See also Identify primary modules.
    """

    variable_revision_number = 2
    category = "Object Processing"
    
    def create_settings(self):
        self.module_name = "IdentifySecondary"
        self.primary_objects = cps.ObjectNameSubscriber("What did you call the primary objects you want to create secondary objects around?","Nuclei")
        self.objects_name = cps.ObjectNameProvider("What do you want to call the objects identified by this module?","Cells")
        self.method = cps.Choice("Select the method to identify the secondary objects (Distance - B uses background; Distance - N does not):",
                                 [M_PROPAGATION, M_WATERSHED_G, M_WATERSHED_I, 
                                  M_DISTANCE_N, M_DISTANCE_B],
                                 M_PROPAGATION)
        self.image_name = cps.ImageNameSubscriber("What did you call the images to be used to find the edges of the secondary objects? For DISTANCE - N, this will not affect object identification, only the final display.",
                                                  "None")
        self.threshold_method = cps.Choice('''Select an automatic thresholding method or choose "Manual" to enter a threshold manually.  To choose a binary image, select "Binary image".  Choosing 'All' will use the Otsu Global method to calculate a single threshold for the entire image group. The other methods calculate a threshold for each image individually. "Set interactively" will allow you to manually adjust the threshold during the first cycle to determine what will work well.''',
                                           [cpthresh.TM_OTSU_GLOBAL,cpthresh.TM_OTSU_ADAPTIVE,cpthresh.TM_OTSU_PER_OBJECT,
                                            cpthresh.TM_MOG_GLOBAL,cpthresh.TM_MOG_ADAPTIVE,cpthresh.TM_MOG_PER_OBJECT,
                                            cpthresh.TM_BACKGROUND_GLOBAL, cpthresh.TM_BACKGROUND_ADAPTIVE, cpthresh.TM_BACKGROUND_PER_OBJECT,
                                            cpthresh.TM_ROBUST_BACKGROUND_GLOBAL, cpthresh.TM_ROBUST_BACKGROUND_ADAPTIVE, cpthresh.TM_ROBUST_BACKGROUND_PER_OBJECT,
                                            cpthresh.TM_RIDLER_CALVARD_GLOBAL, cpthresh.TM_RIDLER_CALVARD_ADAPTIVE, cpthresh.TM_RIDLER_CALVARD_PER_OBJECT,
                                            cpthresh.TM_KAPUR_GLOBAL,cpthresh.TM_KAPUR_ADAPTIVE,cpthresh.TM_KAPUR_PER_OBJECT,
                                            cpthresh.TM_MANUAL, cpthresh.TM_BINARY_IMAGE])
        self.threshold_correction_factor = cps.Float('Threshold correction factor', 1)
        self.threshold_range = cps.FloatRange('Lower and upper bounds on threshold, in the range [0,1]', (0,1),minval=0,maxval=1)
        self.object_fraction = cps.CustomChoice('For MoG thresholding, what is the approximate fraction of image covered by objects?',
                                                ['0.01','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','0.99'])
        self.manual_threshold = cps.Float("What is the manual threshold?",value=0.0,minval=0.0,maxval=1.0)
        self.binary_image = cps.ImageNameSubscriber("What is the binary thresholding image?","None")
        self.distance_to_dilate = cps.Integer("Enter the number of pixels by which to expand the primary objects [Positive integer]",10,minval=1)
        self.regularization_factor = cps.Float("Enter the regularization factor (0 to infinity). Larger=distance,0=intensity)",0.05,minval=0)
        self.use_outlines = cps.Binary("Do you want to save outlines of the images?",False)
        self.outlines_name = cps.OutlineNameProvider("What do you want to call the outlines?","SecondaryOutlines")
        self.two_class_otsu = cps.Choice('Does your image have two classes of intensity value or three?',
                                         [cpmi.O_TWO_CLASS, cpmi.O_THREE_CLASS])
        self.use_weighted_variance = cps.Choice('Do you want to minimize the weighted variance or the entropy?',
                                                [cpmi.O_WEIGHTED_VARIANCE, cpmi.O_ENTROPY])
        self.assign_middle_to_foreground = cps.Choice("Assign pixels in the middle intensity class to the foreground or the background?",
                                                      [cpmi.O_FOREGROUND, cpmi.O_BACKGROUND])
    
    def settings(self):
        return [ self.primary_objects, self.objects_name,   
                 self.method, self.image_name, self.threshold_method, 
                 self.threshold_correction_factor, self.threshold_range,
                 self.object_fraction, self.distance_to_dilate, 
                 self.regularization_factor, self.outlines_name,
                 self.manual_threshold,
                 self.binary_image, self.use_outlines,
                 self.two_class_otsu, self.use_weighted_variance,
                 self.assign_middle_to_foreground ]
    
    def visible_settings(self):
        result = [self.image_name, self.primary_objects, self.objects_name,  
                 self.method]
        if self.method != M_DISTANCE_N:
            result.append(self.threshold_method)
            if self.threshold_method == cpthresh.TM_MANUAL:
                result.append(self.manual_threshold)
            elif self.threshold_method == cpthresh.TM_BINARY_IMAGE:
                result.append(self.binary_image)
            else:
                if self.threshold_algorithm == cpthresh.TM_OTSU:
                    result+= [self.two_class_otsu, self.use_weighted_variance]
                    if self.two_class_otsu == cpmi.O_THREE_CLASS:
                        result.append(self.assign_middle_to_foreground)
            
                result.extend([self.threshold_correction_factor,
                               self.threshold_range, self.object_fraction])
        if self.method in (M_DISTANCE_B,M_DISTANCE_N):
            result.append(self.distance_to_dilate)
        elif self.method == M_PROPAGATION:
            result.append(self.regularization_factor)
        result.append(self.use_outlines)
        if self.use_outlines.value:
            result.append(self.outlines_name)
        return result
    
    def backwards_compatibilize(self,
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
        return setting_values, variable_revision_number, from_matlab

    def run(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_grayscale = True)
        img = image.pixel_data
        mask = image.mask
        objects = workspace.object_set.get_objects(self.primary_objects.value)
        if self.method == M_DISTANCE_N:
            has_threshold = False
        elif self.threshold_method == cpthresh.TM_BINARY_IMAGE:
            binary_image = workspace.image_set.get_image(self.binary_image.value,
                                                         must_be_binary = True)
            local_threshold = numpy.ones(img.shape)
            local_threshold[binary_image.pixel_data] = 0
            global_threshold = otsu(img[mask],
                        self.threshold_range.min,
                        self.threshold_range.max)
            has_threshold = True
        else:
            local_threshold,global_threshold = self.get_threshold(img, mask, None)
            has_threshold = True
        
        if has_threshold:
            thresholded_image = img > local_threshold
        
        labels_in = objects.small_removed_segmented
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
            #
            # Create the final output labels by removing labels in the
            # output matrix that are missing from the segmented image
            # 
            segmented_labels = objects.segmented
            small_removed_segmented_out = labels_out
            segmented_out = self.filter_labels(labels_out, objects)
        elif self.method == M_PROPAGATION:
            labels_out, distance = propagate(img, labels_in, 
                                             thresholded_image,
                                             self.regularization_factor.value)
            small_removed_segmented_out = fill_labeled_holes(labels_out)
            segmented_out = self.filter_labels(small_removed_segmented_out,
                                               objects)
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
            small_removed_segmented_out = fill_labeled_holes(labels_out)
            segmented_out = self.filter_labels(small_removed_segmented_out,
                                               objects)
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
            small_removed_segmented_out = fill_labeled_holes(labels_out)
            segmented_out = self.filter_labels(small_removed_segmented_out,
                                                objects)

        primary_outline = outline(objects.segmented)
        secondary_outline = outline(segmented_out) 
        if workspace.frame != None:
            window_name = "CellProfiler(%s:%d)"%(self.module_name,self.module_num)
            my_frame=cpf.create_or_find(workspace.frame, title="Identify secondary", 
                                        name=window_name, subplots=(2,2))
            title = "Input image, cycle #%d"%(workspace.image_set.number)
            my_frame.subplot_imshow_grayscale(0, 0, img, title)
            my_frame.subplot_imshow_labels(1,0,segmented_out,
                                           "Labeled image")

            secondary_outline_img = img.copy()
            secondary_outline_img[secondary_outline>0] = 1
            outline_img = np.dstack((secondary_outline_img,img,img))
            outline_img.shape=(img.shape[0],img.shape[1],3)
            my_frame.subplot_imshow(0,1, outline_img,"Outlined image")
            
            primary_outline_img = img.copy()
            primary_outline_img[primary_outline>0] = 1
            primary_img = np.dstack((secondary_outline_img,
                                     primary_outline_img,
                                     img))
            primary_img.shape=(img.shape[0],img.shape[1],3)
            my_frame.subplot_imshow(1,1,primary_img,"Primary and output outlines")
            my_frame.Refresh()
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

    def filter_labels(self, labels_out, objects):
        """Filter labels out of the output that are not in the segmented input labels
        
        labels_out - the unfiltered output labels
        objects    - the objects thing, containing both segmented and 
                     small_removed labels
        """
        segmented_labels = objects.segmented
        max_out = np.max(labels_out)
        if max_out > 0:
            lookup = scind.maximum(segmented_labels,labels_out,
                                   range(max_out+1))
            lookup = np.array(lookup, int)
            lookup[0] = 0
            segmented_labels_out = lookup[labels_out]
        else:
            segmented_labels_out = labels_out.copy()
        return segmented_labels_out
        
    def get_measurement_columns(self):
        '''Return column definitions for measurements made by this module'''
        columns = cpmi.get_object_measurement_columns(self.objects_name.value)
        columns += [(self.primary_objects.value,
                     cpmi.FF_CHILDREN_COUNT%self.objects_name.value,
                     cpmeas.COLTYPE_INTEGER),
                    (self.objects_name.value,
                     cpmi.FF_PARENT%self.primary_objects.value,
                     cpmeas.COLTYPE_INTEGER)]
        if self.method != M_DISTANCE_N:
            columns += [(cpmeas.IMAGE, 
                         format % self.objects_name.value,
                         cpmeas.COLTYPE_FLOAT)
                        for format in (cpmi.FF_FINAL_THRESHOLD,
                                       cpmi.FF_ORIG_THRESHOLD,
                                       cpmi.FF_WEIGHTED_VARIANCE,
                                       cpmi.FF_SUM_OF_ENTROPIES)]
        return columns 
