# TODO: Find out why this module makes python segfaults whenever the
# user switches from grayscale to binary.
# TODO: Review new settings with Anne.
# TODO: Update docstring to match the new settings.
# TODO: Test.
__version__="$Revision: 6746 $"

import wx
import matplotlib.cm
import matplotlib.backends.backend_wxagg

from cellprofiler.cpmodule import CPModule
from cellprofiler import cpimage
import cellprofiler.settings as cpsetting
from cellprofiler.gui import cpfigure

from cellprofiler.cpmath.cpmorphology import strel_disk
from scipy.ndimage.morphology import binary_dilation

RETAIN = "Retain"
SHIFT = "Shift"
GRAYSCALE = "Grayscale"
BINARY = "Binary (black and white)"

class ApplyThreshold(CPModule):
    """Pixel intensity below or above a certain threshold is set to zero.

Settings:

When a pixel is thresholded, its intensity value is set to zero so that
it appears black.

If you wish to threshold dim pixels, change the value for which "Pixels
below this value will be set to zero". In this case, the remaining pixels
can retain their original intensity values or are shifted dimmer to
match the threshold used.

If you wish to threshold bright pixels, change the value for which
"Pixels above this value will be set to zero". In this case, you can
expand the thresholding around them by entering the number of pixels to
expand here: This setting is useful to adjust when you are attempting to
exclude bright artifactual objects: you can first set the threshold to
exclude these bright objects, but it may also be desirable to expand the
thresholded region around those bright objects by a certain distance so
as to avoid a 'halo' effect.
"""

    variable_revision_number = 1
    category = "Image Processing"

    def create_variables(self):
        self.module_name = self.__class__.__name__
        self.image_name = cpsetting.NameSubscriber("Which image do you want to threshold?",
                                                   "imagegroup", "None")
        self.thresholded_image_name = cpsetting.NameProvider("What do you want to call the thresholded image?",
                                                             "imagegroup", "ThreshBlue")
        self.binary = cpsetting.Choice("What kind of image would you like to produce?", [GRAYSCALE, BINARY])

        # if not binary:
        self.low = cpsetting.Binary("Set pixels below a certain intensity to zero?", False)
        self.high = cpsetting.Binary("Set pixels above a certain intensity to zero?", False)
        # if not binary and self.low:
        self.low_threshold = cpsetting.Float("Set pixels below this value to zero", 0.0, minval=0, maxval=1)
        self.shift = cpsetting.Binary("Shift the remaining pixels' intensities down by the amount of the threshold?", False)
        # if not binary and self.high:
        self.high_threshold = cpsetting.Float("Set pixels above this value to zero", 1.0, minval=0, maxval=1)
        self.dilation = cpsetting.Float("Number of pixels by which to expand the thresholding around those excluded bright pixels",
                                        0.0)

        # if binary:
        self.binary_threshold = cpsetting.Float("Set pixels below this value to zero and set pixels at least this value to one.",
                                                0.5)

    def visible_variables(self):
        vv = [self.image_name, self.thresholded_image_name, self.binary]
        if self.binary.value == GRAYSCALE:
            vv.append(self.low)
            if self.low.value:
                vv.extend([self.low_threshold, self.shift])
            vv.append(self.high)
            if self.high.value:
                vv.extend([self.high_threshold, self.dilation])
        else:
            vv.append(self.binary_threshold)
        return vv
    
    def variables(self):
        """Return all  variables in a consistent order"""
        return [self.image_name, self.thresholded_image_name,
                self.binary, self.low, self.high, self.low_threshold,
                self.shift, self.high_threshold, self.dilation,
                self.binary_threshold]
    
    def backwards_compatibilize(self, variable_values,
                                variable_revision_number, module_name,
                                from_matlab):
        if from_matlab and variable_revision_number < 4:
            raise NotImplementedError, ("TODO: Handle Matlab CP pipelines for "
                                        "ApplyThreshold with revision < 4")
        if from_matlab and variable_revision_number == 4:
            variable_values = [ variable_values[0],  # ImageName
                                variable_values[1],  # ThresholdedImageName
                                None,
                                None,
                                None,
                                variable_values[5],  # LowThreshold
                                variable_values[6],  # Shift
                                variable_values[7],  # HighThreshold
                                variable_values[8],  # DilationValue
                                variable_values[9],  # BinaryChoice
                                ]
            variable_values[2] = variable_values[9] > 0
            variable_values[3] = LowThreshold > 0
            variable_values[4] = HighThreshold < 1
            variable_revision_number = 1
            from_matlab = False
        return variable_values
        
    def run(self,workspace):
        """Run the module
        
        workspace    - the workspace contains:
            pipeline     - instance of CellProfiler.Pipeline for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - display within this frame (or None to not display)
        """
        input = workspace.image_set.get_image(self.image_name,
                                              must_be_grayscale=True)
        pixels = input.pixel_data.copy()
        if self.binary != 'Grayscale':
            pixels[input.mask] = pixels[input.mask] > self.binary_threshold
        else:
            if self.low.value:
                pixels[input.mask & (pixels < self.low_threshold.value)] = 0
                if self.shift.value:
                    pixels[input.mask] -= self.low_threshold.value
            if self.high.value:
                undilated = input.mask & (pixels >= self.high_threshold.value)
                dilated = binary_dilation(undilated, strel_disk(self.dilation.value), mask=input.mask)
                pixels[dilated] = 0
        output = cpimage.Image(pixels, input.mask)
        workspace.image_set.add(self.thresholded_image_name, output)
        if workspace.display:
            figure = workspace.create_or_find_figure(subplots=(1,2))

            left = figure.subplot(0,0)
            left.clear()
            left.imshow(input.pixel_data,matplotlib.cm.Greys_r)
            left.set_title("Original image: %s"%(self.image_name,))

            right = figure.subplot(0,1)
            right.clear()
            right.imshow(output.pixel_data,matplotlib.cm.Greys_r)
            right.set_title("Thresholded image: %s"%(self.thresholded_image_name,))
