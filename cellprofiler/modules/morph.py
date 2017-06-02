"""
<b>Morph</b> performs low-level morphological operations on binary or grayscale images
<hr>
This module performs a series of morphological operations on a binary image or grayscale image, resulting in an image
of the same type. Many require some image processing knowledge to understand how best to use these morphological
filters in order to achieve the desired result. Note that the algorithms minimize the interference of masked pixels;
for instance, the dilate operation will only consider unmasked pixels in the neighborhood of a pixel when determining
the maximum within that neighborhood.<br>
<br>
The following operations are available:<br>
<br>
<table border="1">
    <tr>
        <td><b>Operation</b></td>
        <td><b>Description</b></td>
        <td><b>Input image type allowed</b></td>
    </tr>
    <tr>
        <td><i>Branchpoints</i></td>
        <td>
            Removes all pixels except those that are the branchpoints of a skeleton. This operation should be applied
            to an image after skeletonizing. It leaves only those pixels that are at the intersection of branches.
            Note that shown here is a window into a hypothetical larger image; the “?” symbols reflect that in this
            small window those values cannot be calculated because the values necessary are outside the window. In
            reality, pixels at the border are handled by padding the image with zeros for almost all operations
            (except it is padded with 1 for hole filling).<br>
            <table>
                <tr>
                    <td>
                        <table border="1">
                            <tr>
                                <td>1</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                    <td>&rarr;</td>
                    <td>
                        <table border="1">
                            <tr>
                                <td>?</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>?</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>?</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Bridge</i></td>
        <td>
            Sets a pixel to 1 if it has two non-zero neighbors that are on opposite sides of this pixel:<br>
            <table>
                <tr>
                    <td>
                        <table border="1">
                            <tr>
                                <td>1</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                    <td>&rarr;</td>
                    <td>
                        <table border="1">
                            <tr>
                                <td>1</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Clean</i></td>
        <td>
            Removes isolated pixels:<br>
            <table>
                <tr>
                    <td>
                        <table border="1">
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                        </table>
                    </td>
                    <td>&rarr;</td>
                    <td>
                        <table border="1">
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Convex hull</i></td>
        <td>Finds the convex hull of a binary image. The convex hull is the smallest convex polygon that fits around
        all foreground pixels of the image: it is the shape that a rubber band would take if stretched around the
        foreground pixels. The convex hull can be used to regularize the boundary of a large, single object in an
        image, for instance, the edge of a well.</td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Diag</i></td>
        <td>
            Fills in pixels whose neighbors are diagnonally connected to 4-connect pixels that are 8-connected:<br>
            <table>
                <tr>
                    <td>
                        <table border="1">
                            <tr>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                        </table>
                    </td>
                    <td>&rarr;</td>
                    <td>
                        <table border="1">
                            <tr>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                    <td>&nbsp;,&nbsp;</td>
                    <td>
                        <table border="1">
                            <tr>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                    <td>&rarr;</td>
                    <td>
                        <table border="1">
                            <tr>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Distance</i></td>
        <td>Computes the distance transform of a binary image. The distance of each foreground pixel is computed to the
        nearest background pixel. The resulting image is then scaled so that the largest distance is 1.</td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Endpoints</i></td>
        <td>
            Removes all pixels except the ones that are at the end of a skeleton.
            Note that shown here is a window into a hypothetical larger image; the “?” symbols reflect that in this
            small window those values cannot be calculated because the values necessary are outside the window. In
            reality, pixels at the border are handled by padding the image with zeros for almost all operations
            (except it is padded with 1 for hole filling).<br>
<br>
            <table>
                <tr>
                    <td>
                        <table border="1">
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                    <td>&rarr;</td>
                    <td>
                        <table border="1">
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>?</td>
                                <td>0</td>
                                <td>0</td>
                                <td>?</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Fill</i></td>
        <td>
            Sets a pixel to 1 if all of its neighbors are 1:<br>
            <table>
                <tr>
                    <td>
                        <table border="1">
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                    <td>&rarr;</td>
                    <td>
                        <table border="1">
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Hbreak</i></td>
        <td>
            Removes pixels that form vertical bridges between horizontal lines:<br>
            <table>
                <tr>
                    <td>
                        <table border="1">
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                    <td>&rarr;</td>
                    <td>
                        <table border="1">
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Majority</i></td>
        <td>
            Each pixel takes on the value of the majority that surround it (keep pixel value to break ties):<br>
            <table>
                <tr>
                    <td>
                        <table border="1">
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                        </table>
                    </td>
                    <td>&rarr;</td>
                    <td>
                        <table border="1">
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Life</i></td>
        <td>
            Applies the interaction rules from the <a href="http://en.wikipedia.org/wiki/Conway%27s_Game_of_Life">Game
            of Life</a>, an example of a cellular automaton.
        </td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>OpenLines</i></td>
        <td>Performs an erosion followed by a dilation using rotating linear structural elements. The effect is to
        return parts of the image that have a linear intensity distribution and suppress dots of the same size.</td>
        <td>Binary, grayscale</td>
    </tr>
    <tr>
        <td><i>Remove</i></td>
        <td>
            Removes pixels that are otherwise surrounded by others (4 connected). The effect is to leave the perimeter
            of a solid object:<br>
            <table>
                <tr>
                    <td>
                        <table border="1">
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                        </table>
                    </td>
                    <td>&rarr;</td>
                    <td>
                        <table border="1">
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Shrink</i></td>
        <td>Performs a thinning operation that erodes unless that operation would change the image's Euler number. This
        means that blobs are reduced to single points and blobs with holes are reduced to rings if shrunken
        indefinitely.</td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>SkelPE</i></td>
        <td>Performs a skeletonizing operation using the metric, PE * D to control the erosion order. PE is the Poisson
        Equation (see Gorelick, "Shape representation and classification using the Poisson Equation", IEEE Transactions
        on Pattern Analysis and Machine Intelligence V28, # 12, 2006) evaluated within the foreground with the boundary
        condition that the background is zero. D is the distance transform (distance of a pixel to the nearest edge).
        The resulting skeleton has fewer spurs but some bit of erosion at the endpoints in the binary image.</td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Spur</i></td>
        <td>
            Removes spur pixels, i.e., pixels that have exactly one 8-connected neighbor. This operation essentially
            removes the endpoints of lines.<br>
            <table>
                <tr>
                    <td>
                        <table border="1">
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                    <td>&rarr;</td>
                    <td>
                        <table border="1">
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>0</td>
                                <td>1</td>
                                <td>0</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Thicken</i></td>
        <td>Dilates the exteriors of objects where that dilation does not 8-connect the object with another. The image
        is labeled and the labeled objects are filled. Unlabeled points adjacent to uniquely labeled points change from
        background to foreground.</td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Thin</i></td>
        <td>Thin lines preserving the Euler number using the thinning algorithm # 1 described in Guo, "Parallel
        Thinning with Two Subiteration Algorithms", <i>Communications of the ACM,</i> Vol 32 #3, page 359. The result
        generally preserves the lines in an image while eroding their thickness.</td>
        <td>Binary</td>
    </tr>
    <tr>
        <td><i>Vbreak</i></td>
        <td>
            Removes pixels that form horizontal bridges between vertical lines:<br>
            <table>
                <tr>
                    <td>
                        <table border="1">
                            <tr>
                                <td>1</td>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>1</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                    <td>&rarr;</td>
                    <td>
                        <table border="1">
                            <tr>
                                <td>1</td>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>0</td>
                                <td>1</td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </td>
        <td>Binary</td>
    </tr>
</table>
"""

import logging
import sys

import numpy as np
import scipy.ndimage as scind

logger = logging.getLogger(__name__)

import cellprofiler.module as cpm
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO
import cellprofiler.image as cpi
import centrosome.cpmorphology as morph
from centrosome.filter import poisson_equation

F_BRANCHPOINTS = 'branchpoints'
F_BRIDGE = 'bridge'
F_CLEAN = 'clean'
F_CONVEX_HULL = 'convex hull'
F_DIAG = 'diag'
F_DISTANCE = 'distance'
F_ENDPOINTS = 'endpoints'
F_FILL = 'fill'
F_HBREAK = 'hbreak'
F_LIFE = 'life'
F_MAJORITY = 'majority'
F_OPENLINES = 'openlines'
F_REMOVE = 'remove'
F_SHRINK = 'shrink'
F_SKELPE = 'skelpe'
F_SPUR = 'spur'
F_THICKEN = 'thicken'
F_THIN = 'thin'
F_VBREAK = 'vbreak'
F_ALL = [F_BRANCHPOINTS, F_BRIDGE, F_CLEAN, F_CONVEX_HULL,
         F_DIAG, F_DISTANCE, F_ENDPOINTS, F_FILL,
         F_HBREAK, F_LIFE, F_MAJORITY, F_OPENLINES, F_REMOVE,
         F_SHRINK, F_SKELPE, F_SPUR, F_THICKEN, F_THIN, F_VBREAK]

R_ONCE = 'Once'
R_FOREVER = 'Forever'
R_CUSTOM = 'Custom'
R_ALL = [R_ONCE, R_FOREVER, R_CUSTOM]

FUNCTION_SETTING_COUNT_V1 = 3
FUNCTION_SETTING_COUNT_V2 = 4
FUNCTION_SETTING_COUNT_V3 = 11
FUNCTION_SETTING_COUNT = 4


class Morph(cpm.Module):
    module_name = "Morph"
    category = "Image Processing"
    variable_revision_number = 5

    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
                "Select the input image", cps.NONE, doc="""
            Select the image that you want to perform a morphological operation on.
            A grayscale image can be
            converted to binary using the <b>ApplyThreshold</b> module. Objects can be
            converted to binary using the <b>ConvertToImage</b> module.""")

        self.output_image_name = cps.ImageNameProvider(
                "Name the output image", "MorphBlue", doc="""
            Enter the name for the output image It will be of the same type as the
            input image.""")

        self.add_button = cps.DoSomething("",
                                          "Add another operation",
                                          self.add_function, doc="""
            Press this button to add an operation that will be applied to the
            image resulting from the previous operation(s). The module repeats
            the previous operation the number of times you select before applying
            the operation added by this button.""")

        self.functions = []
        self.add_function(can_remove=False)

    CUSTOM_REPEATS_TEXT = "Repetition number"
    CUSTOM_REPEATS_DOC = "<i>(Used only if Custom selected)</i><br>Enter the number of times to repeat the operation"

    def add_function(self, can_remove=True):
        group = MorphSettingsGroup()
        group.can_remove = can_remove
        if can_remove:
            group.append("divider", cps.Divider(line=False))
        group.append("function", cps.Choice(
                "Select the operation to perform",
                F_ALL, doc="""
            Choose one of the operations described in this module's help."""))

        group.append("repeats_choice", cps.Choice(
                "Number of times to repeat operation",
                R_ALL, doc="""
            This setting controls the number of times that the same operation is applied
            successively to the image.
            <ul>
            <li><i>%(R_ONCE)s:</i> Perform the operation once on the image.</li>
            <li><i>%(R_FOREVER)s:</i> Perform the operation on the image until successive
            iterations yield the same image.</li>
            <li><i>%(R_CUSTOM)s:</i> Perform the operation a custom number of times.</li>
            </ul>""" % globals()))

        group.append("custom_repeats", cps.Integer(self.CUSTOM_REPEATS_TEXT, 2, 1,
                                                   doc=self.CUSTOM_REPEATS_DOC))

        group.append("rescale_values", cps.Binary(
                "Rescale values from 0 to 1?", True, doc="""
            <i>(Used only for the %(F_DISTANCE)s operation).</i>
            <p>Select <i>%(YES)s</i> to rescale the transformed values to lie between 0 and 1.
            This is the option to use if the distance transformed image is to be used
            for thresholding by an <b>Identify</b> module or the like, which assumes
            a 0-1 scaling.</p>
            <p>Select <i>%(NO)s</i> to leave the values in absolute pixel units.
            This useful in cases where the actual pixel distances are to be used
            downstream as input for a measurement module.</p>
            """ % globals()))

        if can_remove:
            group.append("remove", cps.RemoveSettingButton("", "Remove this operation", self.functions, group))
        self.functions.append(group)

    def prepare_settings(self, setting_values):
        '''Adjust the # of functions to match the # of setting values'''
        assert (len(setting_values) - 2) % FUNCTION_SETTING_COUNT == 0
        function_count = (len(setting_values) - 2) / FUNCTION_SETTING_COUNT
        del self.functions[function_count:]
        while len(self.functions) < function_count:
            self.add_function()

    def settings(self):
        '''Return the settings as saved in the pipeline file'''
        result = [self.image_name, self.output_image_name]
        for function in self.functions:
            result += [function.function, function.repeats_choice, function.custom_repeats, function.rescale_values]
        return result

    def visible_settings(self):
        '''Return the settings as displayed to the user'''
        result = [self.image_name, self.output_image_name]
        for function in self.functions:
            if function.can_remove:
                result.append(function.divider)
            result.append(function.function)
            if function.function == F_DISTANCE:
                result.append(function.rescale_values)
            elif function.function == F_OPENLINES:
                function.custom_repeats.text = "Line length"
                function.custom_repeats.doc = """Only keep lines that have this many pixels or more."""
                result.append(function.custom_repeats)
            elif function.repeats_choice != R_CUSTOM:
                result.append(function.repeats_choice)
            else:
                result.append(function.repeats_choice)
                function.custom_repeats.text = self.CUSTOM_REPEATS_TEXT
                function.custom_repeats.doc = self.CUSTOM_REPEATS_DOC
                result.append(function.custom_repeats)
            if function.can_remove:
                result.append(function.remove)
        result += [self.add_button]
        return result

    def run(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value)
        if image.has_mask:
            mask = image.mask
        else:
            mask = None
        pixel_data = image.pixel_data
        if pixel_data.ndim == 3:
            if any([np.any(pixel_data[:, :, 0] != pixel_data[:, :, plane])
                    for plane in range(1, pixel_data.shape[2])]):
                logger.warn("Image is color, converting to grayscale")
            pixel_data = np.sum(pixel_data, 2) / pixel_data.shape[2]
        for function in self.functions:
            pixel_data = self.run_function(function, pixel_data, mask)
        new_image = cpi.Image(pixel_data, parent_image=image)
        workspace.image_set.add(self.output_image_name.value, new_image)
        if self.show_window:
            workspace.display_data.image = image.pixel_data
            workspace.display_data.pixel_data = pixel_data

    def display(self, workspace, figure):
        image = workspace.display_data.image
        pixel_data = workspace.display_data.pixel_data
        figure.set_subplots((2, 1))
        if pixel_data.dtype.kind == 'b':
            figure.subplot_imshow_bw(0, 0, image,
                                     'Original image: %s' %
                                     self.image_name.value)
            figure.subplot_imshow_bw(1, 0, pixel_data,
                                     self.output_image_name.value,
                                     sharexy=figure.subplot(0, 0))
        else:
            figure.subplot_imshow_grayscale(0, 0, image,
                                            'Original image: %s' %
                                            self.image_name.value)
            figure.subplot_imshow_grayscale(1, 0, pixel_data,
                                            self.output_image_name.value,
                                            sharexy=figure.subplot(0, 0))

    def run_function(self, function, pixel_data, mask):
        '''Apply the function once to the image, returning the result'''
        count = function.repeat_count
        function_name = function.function.value
        custom_repeats = function.custom_repeats.value

        is_binary = pixel_data.dtype.kind == 'b'

        if (function_name in (F_BRANCHPOINTS, F_BRIDGE, F_CLEAN, F_DIAG,
                              F_CONVEX_HULL, F_DISTANCE, F_ENDPOINTS, F_FILL,
                              F_HBREAK, F_LIFE, F_MAJORITY,
                              F_REMOVE, F_SHRINK, F_SKELPE, F_SPUR,
                              F_THICKEN, F_THIN, F_VBREAK)
            and not is_binary):
            # Apply a very crude threshold to the image for binary algorithms
            logger.warning("Warning: converting image to binary for %s\n" %
                           function_name)
            pixel_data = pixel_data != 0

        if function_name in (F_BRANCHPOINTS, F_BRIDGE, F_CLEAN, F_DIAG, F_CONVEX_HULL, F_DISTANCE, F_ENDPOINTS, F_FILL,
                             F_HBREAK, F_LIFE, F_MAJORITY, F_REMOVE, F_SHRINK, F_SKELPE, F_SPUR, F_THICKEN,
                             F_THIN, F_VBREAK, F_OPENLINES):
            # All of these have an iterations argument or it makes no
            # sense to iterate
            if function_name == F_BRANCHPOINTS:
                return morph.branchpoints(pixel_data, mask)
            elif function_name == F_BRIDGE:
                return morph.bridge(pixel_data, mask, count)
            elif function_name == F_CLEAN:
                return morph.clean(pixel_data, mask, count)
            elif function_name == F_CONVEX_HULL:
                if mask is None:
                    return morph.convex_hull_image(pixel_data)
                else:
                    return morph.convex_hull_image(pixel_data & mask)
            elif function_name == F_DIAG:
                return morph.diag(pixel_data, mask, count)
            elif function_name == F_DISTANCE:
                image = scind.distance_transform_edt(pixel_data)
                if function.rescale_values.value:
                    image = image / np.max(image)
                return image
            elif function_name == F_ENDPOINTS:
                return morph.endpoints(pixel_data, mask)
            elif function_name == F_FILL:
                return morph.fill(pixel_data, mask, count)
            elif function_name == F_HBREAK:
                return morph.hbreak(pixel_data, mask, count)
            elif function_name == F_LIFE:
                return morph.life(pixel_data, count)
            elif function_name == F_MAJORITY:
                return morph.majority(pixel_data, mask, count)
            elif function_name == F_OPENLINES:
                return morph.openlines(pixel_data, linelength=custom_repeats, mask=mask)
            elif function_name == F_REMOVE:
                return morph.remove(pixel_data, mask, count)
            elif function_name == F_SHRINK:
                return morph.binary_shrink(pixel_data, count)
            elif function_name == F_SKELPE:
                return morph.skeletonize(
                        pixel_data, mask,
                        scind.distance_transform_edt(pixel_data) *
                        poisson_equation(pixel_data))
            elif function_name == F_SPUR:
                return morph.spur(pixel_data, mask, count)
            elif function_name == F_THICKEN:
                return morph.thicken(pixel_data, mask, count)
            elif function_name == F_THIN:
                return morph.thin(pixel_data, mask, count)
            elif function_name == F_VBREAK:
                return morph.vbreak(pixel_data, mask)
            else:
                raise NotImplementedError("Unimplemented morphological function: %s" %
                                          function_name)
            return pixel_data

    def upgrade_settings(self, setting_values,
                         variable_revision_number, module_name,
                         from_matlab):
        '''Adjust the setting_values of previous revisions to match this one'''
        if from_matlab and variable_revision_number in (1, 2):
            # Settings:
            # image name
            # resulting image name
            # (function, count) repeated 6 times
            new_setting_values = [setting_values[0], setting_values[1]]
            for i in range(6):
                if setting_values[i * 2 + 2] != cps.DO_NOT_USE:
                    new_setting_values.append(setting_values[i * 2 + 2])
                    if (setting_values[i * 2 + 3].isdigit() and
                                int(setting_values[i * 2 + 3]) == 1):
                        new_setting_values += [R_ONCE, "1"]
                    elif setting_values[i * 2 + 3].lower() == "inf":
                        new_setting_values += [R_FOREVER, "2"]
                    else:
                        new_setting_values += [R_CUSTOM, setting_values[i * 2 + 3]]
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 1

        if (not from_matlab) and variable_revision_number == 1:
            new_setting_values = setting_values[:2]
            for i in range(2, len(setting_values), FUNCTION_SETTING_COUNT_V1):
                new_setting_values += setting_values[i:i + FUNCTION_SETTING_COUNT_V1]
                new_setting_values += ["3"]
            setting_values = new_setting_values
            variable_revision_number = 2

        if (not from_matlab) and variable_revision_number == 2:
            new_setting_values = setting_values[:2]
            for i in range(2, len(setting_values), FUNCTION_SETTING_COUNT_V2):
                new_setting_values += setting_values[i:i + FUNCTION_SETTING_COUNT_V2]
                new_setting_values += ["disk", "1", "1", "0", "3", "3", "3,3,111111111"]
            setting_values = new_setting_values
            variable_revision_number = 3

        if (not from_matlab) and variable_revision_number == 3:
            new_setting_values = setting_values[:2]
            for i in range(2, len(setting_values), FUNCTION_SETTING_COUNT_V3):
                new_setting_values += setting_values[i:i + FUNCTION_SETTING_COUNT_V3]
                new_setting_values += [cps.YES]
            setting_values = new_setting_values
            variable_revision_number = 4

        if variable_revision_number == 4:
            functions = setting_values[2::12]

            unavailable = [function for function in functions if function not in F_ALL]

            if len(unavailable) > 0:
                message = "Unsupported operation(s): {}".format(", ".join(unavailable))

                message += "\nYou can reimplement this functionality using equivalent Mathematical Morphology modules."

                raise NotImplementedError(message)

            repeats = setting_values[3::12]

            repeat_counts = setting_values[4::12]

            rescale = setting_values[13::12]

            new_setting_values = list(sum(zip(functions, repeats, repeat_counts, rescale), ()))

            setting_values = setting_values[:2] + new_setting_values

            variable_revision_number = 5

        return setting_values, variable_revision_number, from_matlab


class MorphSettingsGroup(cps.SettingsGroup):
    @property
    def repeat_count(self):
        ''  # of times to repeat'''
        if self.repeats_choice == R_ONCE:
            return 1
        elif self.repeats_choice == R_FOREVER:
            return 10000
        elif self.repeats_choice == R_CUSTOM:
            return self.custom_repeats.value
        else:
            raise ValueError("Unsupported repeat choice: %s" %
                             self.repeats_choice.value)

        '''The thresholding algorithm to run'''
        return self.threshold_method.value.split(' ')[0]
