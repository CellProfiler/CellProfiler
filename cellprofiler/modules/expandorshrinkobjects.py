'''<b>Expand Or Shrink Objects</b> expands or shrinks objects by a defined distance.
<hr>
The module expands or shrinks objects by adding or removing border
pixels. You can specify a certain number of border pixels to be
added or removed, expand objects until they are almost touching or shrink
objects down to a point. Objects are never lost using this module (shrinking 
stops when an object becomes a single pixel). The module can separate touching
objects without otherwise shrinking
the objects.

<p><b>ExpandOrShrinkObjects</b> can perform some specialized morphological operations that 
remove pixels without completely removing an object. See the Settings help (below)
for more detail.</p>

<p><i>Special note on saving images:</i> You can use the settings in this module to pass object
outlines along to the module <b>OverlayOutlines</b> and then save them 
with the <b>SaveImages</b> module. You can also pass the identified objects themselves along to the
object processing module <b>ConvertToImage</b> and then save them with the
<b>SaveImages</b> module.</p>

<h4>Available measurements</h4>
<b>Image measurements:</b>
<ul>
<li><i>Count:</i> Number of expanded/shrunken objects in the image.</li>
</ul>
<b>Object measurements:</b>
<ul>
<li><i>Location_X, Location_Y:</i> Pixel (<i>X,Y</i>) coordinates of the center of mass of 
the expanded/shrunken objects.</li>
</ul>

<p>See also <b>Identify</b> modules.</p>'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2014 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org


import numpy as np
from scipy.ndimage import distance_transform_edt

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
from cellprofiler.settings import YES, NO
import cellprofiler.objects as cpo
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
from cellprofiler.modules.identify import add_object_count_measurements
from cellprofiler.modules.identify import add_object_location_measurements
from cellprofiler.modules.identify import get_object_measurement_columns
from cellprofiler.cpmath.cpmorphology import binary_shrink, thin
from cellprofiler.cpmath.cpmorphology import fill_labeled_holes, adjacent
from cellprofiler.cpmath.cpmorphology import skeletonize_labels, spur
from cellprofiler.cpmath.outline import outline
from cellprofiler.gui.help import RETAINING_OUTLINES_HELP, NAMING_OUTLINES_HELP

O_SHRINK_INF = 'Shrink objects to a point'
O_EXPAND_INF = 'Expand objects until touching'
O_DIVIDE = 'Add partial dividing lines between objects'
O_SHRINK = 'Shrink objects by a specified number of pixels'
O_EXPAND = 'Expand objects by a specified number of pixels'
O_SKELETONIZE = 'Skeletonize each object'
O_SPUR = 'Remove spurs'
O_ALL = [O_SHRINK_INF, O_EXPAND_INF, O_DIVIDE, O_SHRINK, O_EXPAND,
         O_SKELETONIZE, O_SPUR]

class ExpandOrShrinkObjects(cpm.CPModule):

    module_name = 'ExpandOrShrinkObjects'
    category = 'Object Processing'
    variable_revision_number = 1
    def create_settings(self):
        self.object_name = cps.ObjectNameSubscriber(
            "Select the input objects",
            cps.NONE, doc = '''
            Select the objects that you want to expand or shrink.''')
        
        self.output_object_name = cps.ObjectNameProvider(
            "Name the output objects", 
            "ShrunkenNuclei", doc = '''
            Enter a name for the resulting objects.''')
        
        self.operation = cps.Choice(
            "Select the operation",
            O_ALL,  doc = '''
            Select the operation that you want to perform:
            <ul>
            <li><i>%(O_SHRINK_INF)s:</i> Remove all pixels but one from filled objects. Thin objects
            with holes to loops unless the "fill" option is checked.</li>
            <li><i>%(O_EXPAND_INF)s:</i> Expand objects, assigning every pixel in the image to an
            object. Background pixels are assigned to the nearest object.</li>
            <li><i>%(O_DIVIDE)s:</i> Remove pixels from an object that are adjacent to another
            object's pixels unless doing so would change the object's Euler number
            (break an object in two, remove the object completely or open a hole in
            an object).</li>
            <li><i>%(O_SHRINK)s:</i> Remove pixels around the perimeter of an object unless doing
            so would change the object's Euler number (break the object in two, remove the object completely or open
            a hole in the object). You can specify the number of times 
            perimeter pixels should be removed. Processing stops automatically when there are no more
            pixels to remove.</li>
            <li><i>%(O_EXPAND)s:</i> Expand each object by adding background pixels adjacent to the
            image. You can choose the number of times to expand. Processing stops
            automatically if there are no more background pixels.</li>
            <li><i>%(O_SKELETONIZE)s:</i> Erode each object to its skeleton.</li>
            <li><i>%(O_SPUR)s:</i> Remove or reduce the length of spurs in a skeletonized image.
            The algorithm reduces spur size by the number of pixels indicated in the
            setting <i>Number of pixels by which to expand or shrink</i>.</li> 
            </ul>'''%globals())
        
        self.iterations = cps.Integer(
            "Number of pixels by which to expand or shrink", 1, minval=1)
        
        self.wants_fill_holes = cps.Binary(
            "Fill holes in objects so that all objects shrink to a single point?",False, doc="""
            <i>(Used only if one of the "shrink" options selected)</i><br>
            Select <i>%(YES)s</i> to ensure that each object will shrink
            to a single point, by filling the holes in each object.
            <p>Select <i>%(NO)s</i> to preserve the Euler number. in this case, the 
            shrink algorithm preserves each object's Euler number,
            which means that it will erode an object with a hole to a ring in order to
            keep the hole. An object with two holes will be shrunk to two rings
            connected by a line in order to keep from breaking up the object or breaking
            the hole.</p>"""%globals())
        
        self.wants_outlines = cps.Binary(
            "Retain the outlines of the identified objects?",
            False, doc="""
            %(RETAINING_OUTLINES_HELP)s"""%globals())
        
        self.outlines_name = cps.OutlineNameProvider(
            "Name the outline image",
            "ShrunkenNucleiOutlines", doc = """
            %(NAMING_OUTLINES_HELP)s"""%globals())

    def settings(self):
        return [self.object_name, self.output_object_name, self.operation,
                self.iterations, self.wants_fill_holes, self.wants_outlines,
                self.outlines_name]

    def visible_settings(self):
        result = [self.object_name, self.output_object_name, self.operation]
        if self.operation in (O_SHRINK, O_EXPAND, O_SPUR):
            result += [self.iterations]
        if self.operation in (O_SHRINK, O_SHRINK_INF):
            result += [self.wants_fill_holes]
        result += [self.wants_outlines]
        if self.wants_outlines.value:
            result += [self.outlines_name]
        return result

    def run(self, workspace):
        input_objects = workspace.object_set.get_objects(self.object_name.value)
        output_objects = cpo.Objects()
        output_objects.segmented = self.do_labels(input_objects.segmented)
        if (input_objects.has_small_removed_segmented and 
            self.operation not in (O_EXPAND, O_EXPAND_INF, O_DIVIDE)):
            output_objects.small_removed_segmented = \
                self.do_labels(input_objects.small_removed_segmented)
        if (input_objects.has_unedited_segmented and
            self.operation not in (O_EXPAND, O_EXPAND_INF, O_DIVIDE)):
            output_objects.unedited_segmented = \
                self.do_labels(input_objects.unedited_segmented)
        workspace.object_set.add_objects(output_objects,
                                         self.output_object_name.value)
        add_object_count_measurements(workspace.measurements, 
                                      self.output_object_name.value,
                                      np.max(output_objects.segmented))
        add_object_location_measurements(workspace.measurements,
                                         self.output_object_name.value,
                                         output_objects.segmented)
        if self.wants_outlines.value:
            outline_image = cpi.Image(outline(output_objects.segmented) > 0,
                                      parent_image = input_objects.parent_image)
            workspace.image_set.add(self.outlines_name.value, outline_image)

        if self.show_window:
            workspace.display_data.input_objects_segmented = input_objects.segmented
            workspace.display_data.output_objects_segmented = output_objects.segmented

    def display(self, workspace, figure):
        input_objects_segmented = workspace.display_data.input_objects_segmented
        output_objects_segmented = workspace.display_data.output_objects_segmented
        figure.set_subplots((2, 1))
        figure.subplot_imshow_labels(0, 0, input_objects_segmented,
                                     self.object_name.value)
        figure.subplot_imshow_labels(1, 0, output_objects_segmented,
                                     self.output_object_name.value,
                                     sharexy = figure.subplot(0, 0))

    def do_labels(self, labels):
        '''Run whatever transformation on the given labels matrix'''
        if (self.operation in (O_SHRINK, O_SHRINK_INF) and 
            self.wants_fill_holes.value):
            labels = fill_labeled_holes(labels) 
            
        if self.operation == O_SHRINK_INF:
            return binary_shrink(labels)
        elif self.operation == O_SHRINK:
            return binary_shrink(labels, iterations = self.iterations.value)
        elif self.operation in (O_EXPAND, O_EXPAND_INF):
            if self.operation == O_EXPAND_INF:
                distance = np.max(labels.shape)
            else:
                distance = self.iterations.value
            background = labels == 0
            distances, (i,j) = distance_transform_edt(background, 
                                                      return_indices = True)
            out_labels = labels.copy()
            mask = (background & (distances <= distance))
            out_labels[mask] = labels[i[mask],j[mask]]
            return out_labels
        elif self.operation == O_DIVIDE:
            #
            # A pixel must be adjacent to some other label and the object
            # must not disappear.
            #
            adjacent_mask = adjacent(labels)
            thinnable_mask = binary_shrink(labels, 1) != 0
            out_labels = labels.copy()
            out_labels[adjacent_mask & ~ thinnable_mask] = 0
            return out_labels
        elif self.operation == O_SKELETONIZE:
            return skeletonize_labels(labels)
        elif self.operation == O_SPUR:
            return spur(labels, iterations=self.iterations.value)
        else:
            raise NotImplementedError("Unsupported operation: %s" %
                                      self.operation.value)
            

    
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 2:
            inf = setting_values[4] == "Inf"
            if setting_values[3] == "Expand":
                operation = O_EXPAND_INF if inf else O_EXPAND
            elif setting_values[3] == "Shrink":
                operation = (O_SHRINK_INF if inf
                             else O_DIVIDE if setting_values[4] == "0" 
                             else O_SHRINK)
            iterations = "1" if inf else setting_values[4]
            wants_outlines = setting_values[5] != cps.DO_NOT_USE
            setting_values = (setting_values[:2] + 
                              [operation, iterations, cps.NO,
                               cps.YES if wants_outlines else cps.NO,
                               setting_values[5] ])
            from_matlab = False
            variable_revision_number = 1
        return setting_values, variable_revision_number, from_matlab

    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        columns = get_object_measurement_columns(self.output_object_name.value)
        return columns
    
    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        categories = []
        if object_name == cpmeas.IMAGE:
            categories += ["Count"]
        if (object_name == self.output_object_name):
            categories += ("Location","Number")
        return categories
      
    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        result = []
        
        if object_name == cpmeas.IMAGE:
            if category == "Count":
                result += [self.output_object_name.value]
        if object_name == self.output_object_name:
            if category == "Location":
                result += [ "Center_X","Center_Y"]
            elif category == "Number":
                result += ["Object_Number"]
        return result

#
# backwards compatability
#
ExpandOrShrink = ExpandOrShrinkObjects
