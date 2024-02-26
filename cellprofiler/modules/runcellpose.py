import numpy
import os
from skimage.transform import resize
import importlib.metadata

from cellprofiler_core.image import Image
from cellprofiler_core.module.image_segmentation import ImageSegmentation
from cellprofiler_core.object import Objects
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Integer, ImageName, Directory, Filename, Float

CUDA_LINK = "https://pytorch.org/get-started/locally/"
Cellpose_link = " https://doi.org/10.1038/s41592-020-01018-x"
Omnipose_link = "https://doi.org/10.1101/2021.11.03.467199"
cellpose_ver = importlib.metadata.version('cellpose')

__doc__ = f"""\
RunCellpose
===========

**RunCellpose** uses a pre-trained machine learning model (Cellpose) to detect cells or nuclei in an image.

This module is useful for automating simple segmentation tasks in CellProfiler.
The module accepts greyscale input images and produces an object set. Probabilities can also be captured as an image.

Loading in a model will take slightly longer the first time you run it each session. When evaluating
performance you may want to consider the time taken to predict subsequent images.

This module now also supports Ominpose. Omnipose builds on Cellpose, for the purpose of **RunCellpose** it adds 2 additional
features: additional models; bact-omni and cyto2-omni which were trained using the Omnipose architechture, and bact
and the mask reconstruction algorithm for Omnipose that was created to solve over-segemnation of large cells; useful for bacterial cells,
but can be used for other arbitrary and anisotropic shapes. You can mix and match Omnipose models with Cellpose style masking or vice versa.

The module has been updated to be compatible with the latest release of Cellpose. From the old version of the module the 'cells' model corresponds to 'cyto2' model.

Installation:

It is necessary that you have installed Cellpose version >= 1.0.2

You'll want to run `pip install cellpose` on your CellProfiler Python environment to setup Cellpose. If you have an older version of Cellpose
run 'python -m pip install cellpose --upgrade'.

To use Omnipose models, and mask reconstruction method you'll want to install Omnipose 'pip install omnipose' and Cellpose version 1.0.2 'pip install cellpose==1.0.2'.

On the first time loading into CellProfiler, Cellpose will need to download some model files from the internet. This
may take some time. If you want to use a GPU to run the model, you'll need a compatible version of PyTorch and a
supported GPU. Instructions are avaiable at this link: {CUDA_LINK}

Stringer, C., Wang, T., Michaelos, M. et al. Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100–106 (2021). {Cellpose_link}
Kevin J. Cutler, Carsen Stringer, Paul A. Wiggins, Joseph D. Mougous. Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation. bioRxiv 2021.11.03.467199. {Omnipose_link}
============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""


MODEL_NAMES = ['cyto', 'nuclei', 'tissuenet', 'livecell', 'cyto2', 'general',
               'CP', 'CPx', 'TN1', 'TN2', 'TN3', 'LC1', 'LC2', 'LC3', 'LC4',
               'custom']


class RunCellpose(ImageSegmentation):
    category = "Object Processing"

    module_name = "RunCellpose"

    variable_revision_number = 3

    doi = {"Please cite the following when using RunCellPose:": 'https://doi.org/10.1038/s41592-020-01018-x',
    "If you are using Omnipose also cite the following:": 'https://doi.org/10.1101/2021.11.03.467199' }


    def create_settings(self):
        super(RunCellpose, self).create_settings()

        self.expected_diameter = Integer(
            text="Expected object diameter",
            value=15,
            minval=0,
            doc="""\
The average diameter of the objects to be detected. Setting this to 0 will attempt to automatically detect object size.
Note that automatic diameter mode does not work when running on 3D images.

Cellpose models come with a pre-defined object diameter. Your image will be resized during detection to attempt to
match the diameter expected by the model. The default models have an expected diameter of ~16 pixels, if trying to
detect much smaller objects it may be more efficient to resize the image first using the Resize module.
""",
        )

        self.mode = Choice(
            text="Detection mode",
            choices=MODEL_NAMES,
            value='cyto2',
            doc="""\
CellPose comes with models for detecting nuclei or cells. Alternatively, you can supply a custom-trained model
generated using the command line or Cellpose GUI. Custom models can be useful if working with unusual cell types.
""",
        )

        self.omni= Binary(
            text="Use Omnipose for mask reconstruction",
            value=False,
            doc="""\
If enabled, use omnipose mask recontruction features will be used (Omnipose installation required and CellPose >= 1.0)  """
        )

        self.do_3D= Binary(
            text="Use 3D",
            value=False,
            doc="""\
If enabled, 3D specific settings will be available."""
        )

        self.use_gpu = Binary(
            text="Use GPU",
            value=False,
            doc=f"""\
If enabled, Cellpose will attempt to run detection on your system's graphics card (GPU).
Note that you will need a CUDA-compatible GPU and correctly configured PyTorch version, see this link for details:
{CUDA_LINK}

If disabled or incorrectly configured, Cellpose will run on your CPU instead. This is much slower but more compatible
with different hardware setups.

Note that, particularly when in 3D mode, lack of GPU memory can become a limitation. If a model crashes you may need to
re-start CellProfiler to release GPU memory. Resizing large images prior to running them through the model can free up
GPU memory.
"""
        )

        self.use_averaging = Binary(
            text="Use averaging",
            value=True,
            doc="""\
If enabled, CellPose will run it's 4 inbuilt models and take a consensus to determine the results. If disabled, only a
single model will be called to produce results. Disabling averaging is faster to run but less accurate."""
        )

        self.invert = Binary(
            text="Invert images",
            value=False,
            doc="""\
If enabled the image will be inverted and also normalized. For use with fluorescence images using bact model (bact model was trained on phase images"""
        )

        self.supply_nuclei = Binary(
            text="Supply nuclei image as well?",
            value=False,
            doc="""
When detecting whole cells, you can provide a second image featuring a nuclear stain to assist
the model with segmentation. This can help to split touching cells."""
        )

        self.nuclei_image = ImageSubscriber(
            "Select the nuclei image",
            doc="Select the image you want to use as the nuclear stain."
        )

        self.save_probabilities = Binary(
            text="Save probability image?",
            value=False,
            doc="""
If enabled, the probability scores from the model will be recorded as a new image.
Probability >0 is considered as being part of a cell.
You may want to use a higher threshold to manually generate objects.""",
        )

        self.probabilities_name = ImageName(
            "Name the probability image",
            "Probabilities",
            doc="Enter the name you want to call the probability image produced by this module.",
        )

        self.model_directory = Directory(
            "Location of the pre-trained model file",
            doc=f"""\
*(Used only when using a custom pre-trained model)*
Select the location of the pre-trained CellPose model file that will be used for detection."""
        )

        def get_directory_fn():
            """Get the directory for the rules file name"""
            return self.model_directory.get_absolute_path()

        def set_directory_fn(path):
            dir_choice, custom_path = self.model_directory.get_parts_from_path(path)

            self.model_directory.join_parts(dir_choice, custom_path)

        self.model_file_name = Filename(
            "Pre-trained model file name",
            "cyto_0",
            get_directory_fn=get_directory_fn,
            set_directory_fn=set_directory_fn,
            doc=f"""\
*(Used only when using a custom pre-trained model)*
This file can be generated by training a custom model withing the CellPose GUI or command line applications."""
        )

        self.gpu_test = DoSomething(
            "",
            "Test GPU",
            self.do_check_gpu,
            doc=f"""\
Press this button to check whether a GPU is correctly configured.

If you have a dedicated GPU, a failed test usually means that either your GPU does not support deep learning or the
required dependencies are not installed.
If you have multiple GPUs on your system, this button will only test the first one.
""",
        )

        self.flow_threshold = Float(
            text="Flow threshold",
            value=0.4,
            minval=0,
            doc="""\
The flow_threshold parameter is the maximum allowed error of the flows for each mask. The default is flow_threshold=0.4.
Increase this threshold if cellpose is not returning as many masks as you’d expect.
Similarly, decrease this threshold if cellpose is returning too many ill-shaped masks
""",
        )

        self.cellprob_threshold = Float(

            text="Cell probability threshold",
            value=0.0,
            minval=-6.0,
            maxval=6.0,
            doc=f"""\
Cell probability threshold (all pixels with probability above threshold kept for masks). Recommended default is 0.0.
Values vary from -6 to 6
""",
        )

        self.manual_GPU_memory_share = Float(
            text="GPU memory share for each worker",
            value=0.1,
            minval=0.0000001,
            maxval=1,
            doc="""\
Fraction of the GPU memory share available to each worker. Value should be set such that this number times the number
of workers in each copy of CellProfiler times the number of copies of CellProfiler running (if applicable) is <1
""",
        )

        self.stitch_threshold = Float(
            text="Stitch Threshold",
            value=0.0,
            minval=0,
            doc=f"""\
There may be additional differences in YZ and XZ slices that make them unable to be used for 3D segmentation.
In those instances, you may want to turn off 3D segmentation (do_3D=False) and run instead with stitch_threshold>0.
Cellpose will create masks in 2D on each XY slice and then stitch them across slices if the IoU between the mask on the current slice and the next slice is greater than or equal to the stitch_threshold.
""",
        )

        self.min_size = Integer(
            text="Minimum size",
            value=15,
            minval=-1,
            doc="""\
Minimum number of pixels per mask, can turn off by setting value to -1
""",
        )

    def settings(self):
        return [
            self.x_name,
            self.expected_diameter,
            self.mode,
            self.y_name,
            self.use_gpu,
            self.use_averaging,
            self.supply_nuclei,
            self.nuclei_image,
            self.save_probabilities,
            self.probabilities_name,
            self.model_directory,
            self.model_file_name,
            self.flow_threshold,
            self.cellprob_threshold,
            self.manual_GPU_memory_share,
            self.stitch_threshold,
            self.do_3D,
            self.min_size,
            self.omni,
            self.invert,
        ]

    def visible_settings(self):
        if float(cellpose_ver[0:3]) >= 0.6 and int(cellpose_ver[0])<2:
            vis_settings = [self.mode, self.omni, self.x_name]
        else:
            vis_settings = [self.mode, self.x_name]

        if self.mode.value != 'nuclei':
            vis_settings += [self.supply_nuclei]
            if self.supply_nuclei.value:
                vis_settings += [self.nuclei_image]
        if self.mode.value == 'custom':
            vis_settings += [self.model_directory, self.model_file_name,]

        vis_settings += [self.expected_diameter, self.cellprob_threshold, self.min_size, self.flow_threshold, self.y_name, self.invert, self.save_probabilities]

        vis_settings += [self.do_3D, self.stitch_threshold]

        if self.do_3D.value:
            vis_settings.remove( self.stitch_threshold)

        if self.save_probabilities.value:
            vis_settings += [self.probabilities_name]

        vis_settings += [self.use_averaging]

        return vis_settings

    def validate_module(self, pipeline):
        """If using custom model, validate the model file opens and works"""
        if self.mode.value == 'custom':
            model_file = self.model_file_name.value
            model_directory = self.model_directory.get_absolute_path()
            model_path = os.path.join(model_directory, model_file)
            try:
                open(model_path)
            except:
                raise ValidationError(
                    "Failed to load custom file: %s "
                    % model_path, self.model_file_name,
                )
            try:
                from cellpose import models
                model = models.CellposeModel(pretrained_model=model_path, gpu=self.use_gpu.value)
            except:
                raise ValidationError(
                    "Failed to load custom model: %s "
                    % model_path, self.model_file_name,
                )

    def run(self, workspace):
        from cellpose import models
        if self.mode.value != 'custom':
            model = models.Cellpose(model_type= self.mode.value,
                                    gpu=self.use_gpu.value)
        else:
            model_file = self.model_file_name.value
            model_directory = self.model_directory.get_absolute_path()
            model_path = os.path.join(model_directory, model_file)
            model = models.CellposeModel(pretrained_model=model_path, gpu=self.use_gpu.value)

        if self.use_gpu.value and model.torch:
            from torch import cuda
            cuda.set_per_process_memory_fraction(self.manual_GPU_memory_share.value)

        x_name = self.x_name.value
        y_name = self.y_name.value
        images = workspace.image_set
        x = images.get_image(x_name)
        dimensions = x.dimensions
        x_data = x.pixel_data
        anisotropy = 0.0

        if self.do_3D.value:
            anisotropy = x.spacing[0]/x.spacing[1]

        if x.multichannel:
            raise ValueError("Color images are not currently supported. Please provide greyscale images.")

        if self.mode.value != "nuclei" and self.supply_nuclei.value:
            nuc_image = images.get_image(self.nuclei_image.value)
            # CellPose expects RGB, we'll have a blank red channel, cells in green and nuclei in blue.
            if self.do_3D.value:
                x_data = numpy.stack((numpy.zeros_like(x_data), x_data, nuc_image.pixel_data), axis=1)

            else:
                x_data = numpy.stack((numpy.zeros_like(x_data), x_data, nuc_image.pixel_data), axis=-1)

            channels = [2, 3]
        else:
            channels = [0, 0]

        diam = self.expected_diameter.value if self.expected_diameter.value > 0 else None

        try:
            if float(cellpose_ver[0:3]) >= 0.7 and int(cellpose_ver[0])<2:
                y_data, flows, *_ = model.eval(
                    x_data,
                    channels=channels,
                    diameter=diam,
                    net_avg=self.use_averaging.value,
                    do_3D=self.do_3D.value,
                    anisotropy=anisotropy,
                    flow_threshold=self.flow_threshold.value,
                    cellprob_threshold=self.cellprob_threshold.value,
                    stitch_threshold=self.stitch_threshold.value,
                    min_size=self.min_size.value,
                    omni=self.omni.value,
                    invert=self.invert.value,
            )
            else:
                y_data, flows, *_ = model.eval(
                    x_data,
                    channels=channels,
                    diameter=diam,
                    net_avg=self.use_averaging.value,
                    do_3D=self.do_3D.value,
                    anisotropy=anisotropy,
                    flow_threshold=self.flow_threshold.value,
                    cellprob_threshold=self.cellprob_threshold.value,
                    stitch_threshold=self.stitch_threshold.value,
                    min_size=self.min_size.value,
                    invert=self.invert.value,
            )

            y = Objects()
            y.segmented = y_data

        except Exception as a:
                    print(f"Unable to create masks. Check your module settings. {a}")
        finally:
            if self.use_gpu.value and model.torch:
                # Try to clear some GPU memory for other worker processes.
                try:
                    cuda.empty_cache()
                except Exception as e:
                    print(f"Unable to clear GPU memory. You may need to restart CellProfiler to change models. {e}")

        y.parent_image = x.parent_image
        objects = workspace.object_set
        objects.add_objects(y, y_name)

        if self.save_probabilities.value:
            # Flows come out sized relative to CellPose's inbuilt model size.
            # We need to slightly resize to match the original image.
            size_corrected = resize(flows[2], y_data.shape)
            prob_image = Image(
                size_corrected,
                parent_image=x.parent_image,
                convert=False,
                dimensions=len(size_corrected.shape),
            )

            workspace.image_set.add(self.probabilities_name.value, prob_image)

            if self.show_window:
                workspace.display_data.probabilities = size_corrected

        self.add_measurements(workspace)

        if self.show_window:
            if x.volumetric:
                # Can't show CellPose-accepted colour images in 3D
                workspace.display_data.x_data = x.pixel_data
            else:
                workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data
            workspace.display_data.dimensions = dimensions

    def display(self, workspace, figure):
        if self.save_probabilities.value:
            layout = (2, 2)
        else:
            layout = (2, 1)

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions, subplots=layout
        )

        figure.subplot_imshow(
            colormap="gray",
            image=workspace.display_data.x_data,
            title="Input Image",
            x=0,
            y=0,
        )

        figure.subplot_imshow_labels(
            image=workspace.display_data.y_data,
            sharexy=figure.subplot(0, 0),
            title=self.y_name.value,
            x=1,
            y=0,
        )
        if self.save_probabilities.value:
            figure.subplot_imshow(
                colormap="gray",
                image=workspace.display_data.probabilities,
                sharexy=figure.subplot(0, 0),
                title=self.probabilities_name.value,
                x=0,
                y=1,
            )

    def do_check_gpu(self):
        from cellpose import core
        import importlib.util
        torch_installed = importlib.util.find_spec('torch') is not None
        #if the old version of cellpose <2.0, then use istorch kwarg
        if float(cellpose_ver[0:3]) >= 0.7 and int(cellpose_ver[0])<2:
            GPU_works = core.use_gpu(istorch=torch_installed)
        else: #if new version of cellpose, use use_torch kwarg
            GPU_works = core.use_gpu(use_torch=torch_installed)
        if GPU_works:
            message = "GPU appears to be working correctly!"
        else:
            message = "GPU test failed. There may be something wrong with your configuration."
        import wx
        wx.MessageBox(message, caption="GPU Test")


    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            setting_values = setting_values+["0.4", "0.0"]
            variable_revision_number = 2
        if variable_revision_number == 2:
            setting_values = setting_values + ["0.0", False, "15", "1.0", False, False]
            variable_revision_number = 3
        return setting_values, variable_revision_number
