#################################
#
# Imports from useful Python libraries
#
#################################

import numpy
import os
import skimage
import importlib.metadata
import subprocess
import uuid
import shutil
import logging
import sys

#################################
#
# Imports from CellProfiler
#
##################################

from cellprofiler_core.image import Image
from cellprofiler_core.module.image_segmentation import ImageSegmentation
from cellprofiler_core.object import Objects
from cellprofiler_core.setting import Binary, ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.preferences import get_default_output_directory
from cellprofiler_core.setting.text import (
    Integer,
    ImageName,
    Directory,
    Filename,
    Float,
)

CUDA_LINK = "https://pytorch.org/get-started/locally/"
Cellpose_link = " https://doi.org/10.1038/s41592-020-01018-x"
Omnipose_link = "https://doi.org/10.1101/2021.11.03.467199"
LOGGER = logging.getLogger(__name__)

__doc__ = f"""\
RunCellpose
===========

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""
#
# Constants
#
# It's good programming practice to replace things like strings with
# constants if they will appear more than once in your program. That way,
# if someone wants to change the text, that text will change everywhere.
# Also, you can't misspell it by accident.
#

DOCKER_NAME = "cellprofiler/runcellpose_no_pretrained:0.1"

SOME_CHOICES = ['cyto','nuclei','tissuenet','livecell', 'cyto2', 'general',
                'CP', 'CPx', 'TN1', 'TN2', 'TN3', 'LC1', 'LC2', 'LC3', 'LC4', 'custom']


class SegmentationTemplate(ImageSegmentation):
    category = "Object Processing"

    module_name = "SegmentationTemplate"

    variable_revision_number = 1

    doi = {
        "Please cite the following when using RunCellPose:": "https://doi.org/10.1038/s41592-020-01018-x",
        "If you are using Omnipose also cite the following:": "https://doi.org/10.1101/2021.11.03.467199",
    }

    def create_settings(self):
        super(SegmentationTemplate, self).create_settings()

        self.docker_or_python = Choice(
            text="Run this segmentation thing in docker or local python environment",
            choices=["Docker", "Python"],
            value="Docker",
            doc="""\
If Docker is selected, ensure that Docker Desktop is open and running on your
computer. On first run of the X plugin, the Docker container will be
downloaded. However, this slow downloading process will only have to happen
once.

If Python is selected, the Python environment in which CellProfiler and Cellpose
are installed will be used.
""",
        )

        self.docker_image = Choice(
            text="Select Cellpose docker image",
            choices=[DOCKER_NAME],
            value=DOCKER_NAME,
            doc="""\
Select which Docker image to use for running Cellpose.

If you are not using a custom model, you can select
**"{CELLPOSE_DOCKER_IMAGE_WITH_PRETRAINED}"**. If you are using a custom model,
you can use either **"{CELLPOSE_DOCKER_NO_PRETRAINED}"** or
**"{CELLPOSE_DOCKER_IMAGE_WITH_PRETRAINED}"**, but the latter will be slightly
larger (~500 MB) due to including all of the pretrained models.
"""
        )

        self.some_numerical_parameter = Integer(
            text="Some configurable parameter",
            value=30,
            minval=0,
            doc="""\
""",
        )

        self.some_listy_parameter = Choice(
            text="Detection mode",
            choices=SOME_CHOICES,
            value=SOME_CHOICES[0],
            doc="""
""",
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
""",
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

    def settings(self):
        return [
            self.x_name,
            self.docker_or_python,
            self.docker_image,
            self.some_listy_parameter,
            self.some_numerical_parameter,
            self.y_name,
            self.use_gpu,
        ]

    def visible_settings(self):
        vis_settings = [self.docker_or_python]

        if self.docker_or_python.value == "Docker":
            vis_settings += [self.docker_image]


        vis_settings += [
            self.some_numerical_parameter,
            self.some_listy_parameter,
            self.y_name,
        ]

        if self.docker_or_python.value == 'Python':
            if self.use_gpu.value:
                vis_settings += [self.gpu_test]

        return vis_settings


    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value
        images = workspace.image_set
        x = images.get_image(x_name)
        x_data = x.pixel_data


        if self.docker_or_python.value == "Python":

            from cellpose import models, io, core, utils

            from torch import cuda
            cuda.set_per_process_memory_fraction(self.manual_GPU_memory_share.value)

            try:
                y_data, flows, *_ = some_function(uses_our_parameters)
        

            except Exception as a:
                        print(f"Unable to create masks. Check your module settings. {a}")
            finally:
                if self.use_gpu.value and model.torch:
                    # Try to clear some GPU memory for other worker processes.
                    try:
                        cuda.empty_cache()
                    except Exception as e:
                        print(f"Unable to clear GPU memory. You may need to restart CellProfiler to change models. {e}")

        elif self.docker_or_python.value == "Docker":
            # Define how to call docker
            docker_path = "docker" if sys.platform.lower().startswith("win") else "/usr/local/bin/docker"
            # Create a UUID for this run
            unique_name = str(uuid.uuid4())
            # Directory that will be used to pass images to the docker container
            temp_dir = os.path.join(get_default_output_directory(), ".cellprofiler_temp", unique_name)
            temp_img_dir = os.path.join(temp_dir, "img")
            
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(temp_img_dir, exist_ok=True)

            temp_img_path = os.path.join(temp_img_dir, unique_name+".tiff")
            if self.mode.value == "custom":
                model_file = self.model_file_name.value
                model_directory = self.model_directory.get_absolute_path()
                model_path = os.path.join(model_directory, model_file)
                temp_model_dir = os.path.join(temp_dir, "model")

                os.makedirs(temp_model_dir, exist_ok=True)
                # Copy the model
                shutil.copy(model_path, os.path.join(temp_model_dir, model_file))

            # Save the image to the Docker mounted directory
            skimage.io.imsave(temp_img_path, x_data)

            cmd = f"""
            {docker_path} run --rm -v {temp_dir}:/data
            {self.docker_image.value}
            {some_flags}
            """

            try:
                subprocess.run(cmd.split(), text=True)
                segmentation_output = numpy.load(os.path.join(temp_img_dir, unique_name + "_seg.npy"), allow_pickle=True).item()

                y_data = segmentation_output["masks"]
                flows = segmentation_output["flows"]
            finally:      
                # Delete the temporary files
                try:
                    shutil.rmtree(temp_dir)
                except:
                    LOGGER.error("Unable to delete temporary directory, files may be in use by another program.")
                    LOGGER.error("Temp folder is subfolder {tempdir} in your Default Output Folder.\nYou may need to remove it manually.")


        y = Objects()
        y.segmented = y_data
        y.parent_image = x.parent_image
        objects = workspace.object_set
        objects.add_objects(y, y_name)


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

    def do_check_gpu(self):
        import importlib.util
        torch_installed = importlib.util.find_spec('torch') is not None
        #if the old version of cellpose <2.0, then use istorch kwarg
        if float(self.cellpose_ver[0:3]) >= 0.7 and int(self.cellpose_ver[0])<2:
            GPU_works = core.use_gpu(istorch=torch_installed)
        else:  # if new version of cellpose, use use_torch kwarg
            GPU_works = core.use_gpu(use_torch=torch_installed)
        if GPU_works:
            message = "GPU appears to be working correctly!"
        else:
            message = (
                "GPU test failed. There may be something wrong with your configuration."
            )
        import wx

        wx.MessageBox(message, caption="GPU Test")

