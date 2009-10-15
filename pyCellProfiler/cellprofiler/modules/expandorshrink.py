'''<b>ExpandOrShrink</b> - Expands or shrinks objects by a defined distance
<hr>
The module expands or shrinks objects by adding or removing border
pixels. You can specify a certain number of border pixels to be
added or removed, expand objects until they are almost touching or shrink
objects down to a point. Objects are never lost using this module (shrinking 
stops when an object becomes a single pixel). The module can separate touching
objects (which can be created by <b>IdentifySecondary</b>) without otherwise shrinking
the objects.

ExpandOrShrink can perform some specialized morphological operations that 
remove pixels without completely removing an object.  See the settings (below)
for more detail.

Special note on saving images: Using the settings in this module, object
outlines can be passed along to the module <b>OverlayOutlines</b> and then saved
with the <b>SaveImages</b> module. Objects themselves can be passed along to the
object processing module <b>ConvertToImage</b> and then saved with the
SaveImages module.

See also <b>Identify</b> modules.'''
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

import numpy as np
from scipy.ndimage import distance_transform_edt

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.objects as cpo
from cellprofiler.cpmath.cpmorphology import binary_shrink, thin
from cellprofiler.cpmath.cpmorphology import fill_labeled_holes, adjacent
from cellprofiler.cpmath.cpmorphology import skeletonize_labels, spur
from cellprofiler.cpmath.outline import outline

O_SHRINK_INF = 'Shrink objects to a point'
O_EXPAND_INF = 'Expand objects until touching'
O_DIVIDE = 'Add partial dividing lines between objects'
O_SHRINK = 'Shrink objects by a specified number of pixels'
O_EXPAND = 'Expand objects by a specified number of pixels'
O_SKELETONIZE = 'Skeletonize each object'
O_SPUR = 'Remove spurs'
O_ALL = [O_SHRINK_INF, O_EXPAND_INF, O_DIVIDE, O_SHRINK, O_EXPAND,
         O_SKELETONIZE, O_SPUR]

DOC_FILL_HOLES = '''The shrink algorithm preserves each object's Euler number
which means that it will erode an object with a hole to a ring in order to
keep the hole and it will erode an object with two holes to two rings
connected by a line in order to keep from breaking up the object or breaking
the hole. If you fill the holes in each object, then each object will shrink
to a single point.'''

class ExpandOrShrink(cpm.CPModule):

    category = 'Object Processing'
    variable_revision_number = 1
    def create_settings(self):
        self.module_name = 'ExpandOrShrink'
        self.object_name = cps.ObjectNameSubscriber("Select the input objects",
                                                    "None", doc = '''What did you call the objects you want to expand or shrink?''')
        self.output_object_name = cps.ObjectNameProvider("Name the output objects", 
                                                         "ShrunkenNuclei", doc = '''What do you want to call the resulting objects?''')
        self.operation = cps.Choice("What operation do you want to perform?",
                                    O_ALL,  doc = '''
                                    <ul><li>Shrink objects to a point: Remove all pixels but one from filled objects. Thin objects
                                    with holes to loops unless the "fill" option is checked.</li>
                                    <li>Expand objects until touching: Expand objects, assigning every pixel in the image to an
                                    object. Background pixels are assigned to the nearest object.</li>
                                    <li>Add partial dividing lines between objects: Remove pixels from an object that are adjacent to another
                                    object's pixels unless doing so would change the object's Euler number
                                    (break an object in two, remove the object completely or open a hole in
                                    an object).</li>
                                    <li>Shrink objects by a specified number of pixels: Remove pixels around the perimeter of an object unless doing
                                    so would break the object in two, remove the object completely or open
                                    a hole in the object. The user can choose the number of times to remove
                                    perimeter pixels. Processing stops automatically when there are no more
                                    pixels to remove.</li>
                                    <li>Expand objects by a specified number of pixels: Expand each object by adding background pixels adjacent to the
                                    image. The user can choose the number of times to expand. Processing stops
                                    automatically if there are no more background pixels.</li>
                                    <li>Skeletonize each object: Erode each object to its skeleton.</li>
                                    <li>Remove spurs: Remove or reduce the length of spurs in a skeletonized image.
                                    The algorithm reduces spur size by the number of pixels indicated in the
                                    setting "Enter the number of pixels by which to expand or shrink."</li> </ul>              
                                    ''')
        self.iterations = cps.Integer("Number of pixels by which to expand or shrink",
                                      1, minval=1)
        self.wants_fill_holes = cps.Binary("Do you want to fill holes in objects so that all objects shrink to a single point?",
                                           False, doc=DOC_FILL_HOLES)
        self.wants_outlines = cps.Binary("Do you want to save the outlines of the identified objects",
                                         False)
        self.outlines_name = cps.OutlineNameProvider("What do you want to call the outlines of the identified objects?",
                                                     "ShrunkenNucleiOutlines")

    def settings(self):
        return [self.object_name, self.output_object_name, self.operation,
                self.iterations, self.wants_fill_holes, self.wants_outlines,
                self.outlines_name]

    def backwards_compatibilize(self, setting_values, variable_revision_number, 
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
        
        if self.wants_outlines.value:
            outlines = outline(output_objects.segmented)
            workspace.add_outline(self.outlines_name.value, outlines)
        
        if workspace.frame is not None:
            figure = workspace.create_or_find_figure(subplots=(1,2))
            figure.subplot_imshow_labels(0,0,input_objects.segmented,
                                         self.object_name.value)
            figure.subplot_imshow_labels(0,1,output_objects.segmented,
                                         self.output_object_name.value)
    
    def do_labels(self, labels):
        '''Run whatever transformation on the given labels matrix'''
        if (self.operation in (O_SHRINK, O_SHRINK_INF) and 
            self.wants_fill_holes.value):
            labels = fill_labeled_holes(labels) 
            
        if self.operation == O_SHRINK_INF:
            return binary_shrink(labels)
        elif self.operation == O_SHRINK:
            return thin(labels, iterations = self.iterations.value)
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
            # A pixel must be adjacent to some other label and thinnable
            # in order to be eliminated.
            #
            adjacent_mask = adjacent(labels)
            thinnable_mask = thin(labels, iterations=1) != labels
            out_labels = labels.copy()
            out_labels[adjacent_mask & thinnable_mask] = 0
            return out_labels
        elif self.operation == O_SKELETONIZE:
            return skeletonize_labels(labels)
        elif self.operation == O_SPUR:
            return spur(labels, iterations=self.iterations.value)
        else:
            raise NotImplementedError("Unsupported operation: %s" %
                                      self.operation.value)
            

    