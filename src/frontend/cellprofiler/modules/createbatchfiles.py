"""
CreateBatchFiles
================

**CreateBatchFiles** produces files that allow individual batches of
images to be processed separately on a cluster of computers.

This module creates files that can be submitted in parallel to a cluster
for faster processing. It should be placed at the end of an image
processing pipeline.

If your computer mounts the file system differently than the cluster
computers, **CreateBatchFiles** can replace the necessary parts of the
paths to the image and output files.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============
"""

import logging
import os
import re
import sys
import zlib
import numpy

from packaging.version import Version

from cellprofiler_core.constants.measurement import F_BATCH_DATA_H5
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.preferences import get_absolute_path
from cellprofiler_core.preferences import get_default_image_directory
from cellprofiler_core.preferences import get_default_output_directory
from cellprofiler_core.preferences import get_headless
from cellprofiler_core.preferences import set_default_image_directory
from cellprofiler_core.preferences import set_default_output_directory
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import Divider
from cellprofiler_core.setting import Setting
from cellprofiler_core.setting import SettingsGroup
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.do_something import DoSomething
from cellprofiler_core.setting.do_something import RemoveSettingButton
from cellprofiler_core.setting.text import Text, Integer
from cellprofiler_core.workspace import Workspace

from cellprofiler import __version__ as cellprofiler_version

LOGGER = logging.getLogger(__name__)

"""# of settings aside from the mappings"""
S_FIXED_COUNT = 8
"""# of settings per mapping"""
S_PER_MAPPING = 2


class CreateBatchFiles(Module):
    #
    # How it works:
    #
    # There are three hidden settings: batch_mode, pickled_image_set_list, and
    #     distributed_mode
    # batch_mode controls the mode: False means "save the pipeline" and
    #     True means "run the pipeline"
    # pickled_image_set_list holds the state of the image set list. If
    #     batch_mode is False, we save the state of the image set list in
    #     pickled_image_set_list. If batch_mode is True, we load the state
    #     from pickled_image_set_list.
    # distributed_mode indicates whether the pipeline is being
    #     processed by distributed workers, in which case, the default
    #     input and output directories are set to the temporary
    #     directory.
    module_name = "CreateBatchFiles"
    category = "File Processing"
    variable_revision_number = 8

    def volumetric(self):
        return True

    #
    def create_settings(self):
        """Create the module settings and name the module"""
        self.wants_default_output_directory = Binary(
            "Store batch files in default output folder?",
            True,
            doc="""\
Select "*Yes*" to store batch files in the Default Output folder.
Select "*No*" to enter the path to the folder that will be used to
store these files. The Default Output folder can be set by clicking the "View output settings" button in the main CP window, or in CellProfiler Preferences. """
            % globals(),
        )

        self.custom_output_directory = Text(
            "Output folder path",
            get_default_output_directory(),
            doc="Enter the path to the output folder. (Used only if not using the default output folder)",
        )

        # Worded this way not because I am windows-centric but because it's
        # easier than listing every other OS in the universe except for VMS
        self.remote_host_is_windows = Binary(
            "Are the cluster computers running Windows?",
            False,
            doc="""\
Select "*Yes*" if the cluster computers are running one of the
Microsoft Windows operating systems. In this case, **CreateBatchFiles**
will modify all paths to use the Windows file separator (backslash \\\\ ).
Select "*No*" for **CreateBatchFiles** to modify all paths to use the
Unix or Macintosh file separator (slash / )."""
            % globals(),
        )

        self.batch_mode = Binary("Hidden- in batch mode", False)
        self.distributed_mode = Binary("Hidden- in distributed mode", False)
        self.default_image_directory = Setting(
            "Hidden- default input folder at time of save",
            get_default_image_directory(),
        )
        self.revision = Integer("Hidden- revision number", 0)
        self.from_old_matlab = Binary("Hidden- from old matlab", False)
        self.acknowledge_old_matlab = DoSomething(
            "Could not update CP1.0 pipeline to be compatible with CP2.0.  See module notes.",
            "OK",
            self.clear_old_matlab,
        )
        self.mappings = []
        self.add_mapping()
        self.add_mapping_button = DoSomething(
            "",
            "Add another path mapping",
            self.add_mapping,
            doc="""\
Use this option if another path must be mapped because there is a difference
between how the local computer sees a folder location vs. how the cluster
computer sees the folder location.""",
        )

    def add_mapping(self):
        group = SettingsGroup()
        group.append(
            "local_directory",
            Text(
                "Local root path",
                get_default_image_directory(),
                doc="""\
Enter the path to files on this computer. This is the root path on the
local machine (i.e., the computer setting up the batch files).

For instance, a Windows machine might access files images by mounting the file system using a drive
letter, like this:

``Z:\your_data\images``

and the cluster computers access the same file system like this:

``/server_name/your_name/your_data/images``

In this case, since the ``your_data\images`` portion of the path is
the same for both, the local root path is the portion prior, i.e.,
``Z:\`` and similarly for the cluster root path, i.e.,
``/server_name/your_name/``.

If **CreateBatchFiles** finds any pathname that matches the local root path
at the beginning, it will replace that matching portion with the cluster root path.

For example, if you have mapped the remote cluster machine like this:

``Z:\your_data\images``

(on a Windows machine, for instance) and the cluster machine sees the same folder like this:

``/server_name/your_name/your_data/images``

you would enter ``Z:\`` here for the local root path and ``/server_name/your_name/`` for the
cluster root path in the next setting.""",
            ),
        )

        group.append(
            "remote_directory",
            Text(
                "Cluster root path",
                get_default_image_directory(),
                doc="""\
Enter the path to files on the cluster. This is the cluster root path,
i.e., how the cluster machine sees the top-most folder where your
input/output files are stored.

For instance, a Windows machine might access files images by mounting the file system using a drive
letter, like this:

``Z:\your_data\images``

and the cluster computers access the same file system like this:

``/server_name/your_name/your_data/images``

In this case, since the ``your_data\images`` portion of the path is
the same for both, the local root path is the portion prior, i.e.,
``Z:\`` and similarly for the cluster root path, i.e.,
``/server_name/your_name/``.

If **CreateBatchFiles** finds any pathname that matches the local root path
at the beginning, it will replace that matching portion with the cluster root path.

For example, if you have mapped the remote cluster machine like this:

``Z:\your_data\images``

(on a Windows machine, for instance) and the cluster machine sees the same folder like this:

``/server_name/your_name/your_data/images``

you would enter ``Z:\`` in the previous setting for the local root
path and ``/server_name/your_name/`` here for the cluster root path.""",
            ),
        )
        group.append(
            "remover",
            RemoveSettingButton("", "Remove this path mapping", self.mappings, group),
        )
        group.append("divider", Divider(line=False))
        self.mappings.append(group)

    def settings(self):
        result = [
            self.wants_default_output_directory,
            self.custom_output_directory,
            self.remote_host_is_windows,
            self.batch_mode,
            self.distributed_mode,
            self.default_image_directory,
            self.revision,
            self.from_old_matlab,
        ]
        for mapping in self.mappings:
            result += [mapping.local_directory, mapping.remote_directory]
        return result

    def prepare_settings(self, setting_values):
        if (len(setting_values) - S_FIXED_COUNT) % S_PER_MAPPING != 0:
            raise ValueError(
                "# of mapping settings (%d) "
                "is not a multiple of %d"
                % (len(setting_values) - S_FIXED_COUNT, S_PER_MAPPING)
            )
        mapping_count = (len(setting_values) - S_FIXED_COUNT) / S_PER_MAPPING
        while mapping_count < len(self.mappings):
            del self.mappings[-1]

        while mapping_count > len(self.mappings):
            self.add_mapping()

    def visible_settings(self):
        if self.from_old_matlab:
            return [self.acknowledge_old_matlab]
        result = [self.wants_default_output_directory]
        if not self.wants_default_output_directory.value:
            result += [self.custom_output_directory]
        result += [self.remote_host_is_windows]
        for mapping in self.mappings:
            result += mapping.visible_settings()
        result += [self.add_mapping_button]
        return result

    def help_settings(self):
        help_settings = [
            self.wants_default_output_directory,
            self.custom_output_directory,
            self.remote_host_is_windows,
        ]
        for mapping in self.mappings:
            help_settings += [mapping.local_directory, mapping.remote_directory]

        return help_settings

    def prepare_run(self, workspace):
        """Invoke the image_set_list pickling mechanism and save the pipeline"""

        pipeline = workspace.pipeline
        image_set_list = workspace.image_set_list

        if pipeline.test_mode or self.from_old_matlab:
            return True
        if self.batch_mode.value:
            self.enter_batch_mode(workspace)
            return True
        else:
            path = self.save_pipeline(workspace)
            if not get_headless():
                import wx

                wx.MessageBox(
                    "CreateBatchFiles saved pipeline to %s" % path,
                    caption="CreateBatchFiles: Batch file saved",
                    style=wx.OK | wx.ICON_INFORMATION,
                )
            return False

    def run(self, workspace):
        # all the actual work is done in prepare_run
        pass

    def clear_old_matlab(self):
        self.from_old_matlab.value = "No"

    def validate_module(self, pipeline):
        """Make sure the module settings are valid"""
        # Ensure we're not an un-updatable version of the module from way back.
        if self.from_old_matlab.value:
            raise ValidationError(
                "The pipeline you loaded was from an old version of CellProfiler 1.0, "
                "which could not be made compatible with this version of CellProfiler.",
                self.acknowledge_old_matlab,
            )
        # This must be the last module in the pipeline
        if id(self) != id(pipeline.modules()[-1]):
            raise ValidationError(
                "The CreateBatchFiles module must be " "the last in the pipeline.",
                self.wants_default_output_directory,
            )

    def validate_module_warnings(self, pipeline):
        """Warn user re: Test mode """
        if pipeline.test_mode:
            raise ValidationError(
                "CreateBatchFiles will not produce output in Test Mode",
                self.wants_default_output_directory,
            )

    def save_pipeline(self, workspace, outf=None):
        """Save the pipeline in Batch_data.mat

        Save the pickled image_set_list state in a setting and put this
        module in batch mode.

        if outf is not None, it is used as a file object destination.
        """
        if outf is None:
            if self.wants_default_output_directory.value:
                path = get_default_output_directory()
            else:
                path = get_absolute_path(self.custom_output_directory.value)
                os.makedirs(path, exist_ok=True)
            h5_path = os.path.join(path, F_BATCH_DATA_H5)
        else:
            h5_path = outf

        image_set_list = workspace.image_set_list
        pipeline = workspace.pipeline
        m = Measurements(copy=workspace.measurements, filename=h5_path)
        try:
            assert isinstance(pipeline, Pipeline)
            assert isinstance(m, Measurements)

            orig_pipeline = pipeline
            pipeline = pipeline.copy()
            # this use of workspace.frame is okay, since we're called from
            # prepare_run which happens in the main wx thread.
            target_workspace = Workspace(
                pipeline, None, None, None, m, image_set_list, workspace.frame
            )
            pipeline.prepare_to_create_batch(target_workspace, self.alter_path)
            bizarro_self = pipeline.module(self.module_num)
            ver = Version(cellprofiler_version)
            bizarro_self.revision.value = int(f"{ver.major}{ver.minor}{ver.micro}")
            if self.wants_default_output_directory:
                bizarro_self.custom_output_directory.value = self.alter_path(
                    get_default_output_directory()
                )
            bizarro_self.default_image_directory.value = self.alter_path(
                get_default_image_directory()
            )
            bizarro_self.batch_mode.value = True
            pipeline.write_pipeline_measurement(m)
            orig_pipeline.write_pipeline_measurement(m, user_pipeline=True)
            #
            # Write the path mappings to the batch measurements
            #
            m.write_path_mappings(
                [
                    (mapping.local_directory.value, mapping.remote_directory.value)
                    for mapping in self.mappings
                ]
            )
            return h5_path
        finally:
            m.close()

    def is_create_batch_module(self):
        return True

    def in_batch_mode(self):
        """Tell the system whether we are in batch mode on the cluster"""
        return self.batch_mode.value

    def enter_batch_mode(self, workspace):
        """Restore the image set list from its setting as we go into batch mode"""
        pipeline = workspace.pipeline
        assert isinstance(pipeline, Pipeline)
        assert not self.distributed_mode, "Distributed mode no longer supported"
        default_output_directory = self.custom_output_directory.value
        default_image_directory = self.default_image_directory.value
        if os.path.isdir(default_output_directory):
            set_default_output_directory(default_output_directory)
        else:
            LOGGER.info(
                'Batch file default output directory, "%s", does not exist'
                % default_output_directory
            )
        if os.path.isdir(default_image_directory):
            set_default_image_directory(default_image_directory)
        else:
            LOGGER.info(
                'Batch file default input directory "%s", does not exist'
                % default_image_directory
            )

    def turn_off_batch_mode(self):
        """Remove any indications that we are in batch mode

        This call restores the module to an editable state.
        """
        self.batch_mode.value = False
        self.batch_state = numpy.zeros((0,), numpy.uint8)

    def alter_path(self, path, **varargs):
        """Modify the path passed so that it can be executed on the remote host

        path = path to modify
        regexp_substitution - if true, exclude \g<...> from substitution
        """
        regexp_substitution = varargs.get("regexp_substitution", False)
        for mapping in self.mappings:
            local_directory = mapping.local_directory.value
            remote_directory = mapping.remote_directory.value
            if regexp_substitution:
                local_directory = local_directory.replace("\\", "\\\\")
                remote_directory = remote_directory.replace("\\", "\\\\")

            if sys.platform.startswith("win"):
                # Windows is case-insensitive so do case-insensitive mapping
                if path.upper().startswith(local_directory.upper()):
                    path = remote_directory + path[len(local_directory) :]
            else:
                if path.startswith(local_directory):
                    path = remote_directory + path[len(local_directory) :]
        if self.remote_host_is_windows.value:
            path = path.replace("/", "\\")
        elif regexp_substitution:
            path = re.subn("\\\\\\\\", "/", path)[0]
            path = re.subn("\\\\(?!g<[^>]*>)", "/", path)[0]
        else:
            path = path.replace("\\", "/")
        return path

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            setting_values = (
                setting_values[:5]
                + [get_default_image_directory()]
                + setting_values[5:]
            )
            variable_revision_number = 2
        if variable_revision_number == 2:
            ver = Version(cellprofiler_version)
            setting_values = (
                setting_values[:6]
                + [int(f"{ver.major}{ver.minor}{ver.micro}")]
                + setting_values[6:]
            )
            variable_revision_number = 3
        if variable_revision_number == 3:
            # Pickled image list is now the batch state
            self.batch_state = numpy.array(zlib.compress(setting_values[4]))
            setting_values = setting_values[:4] + setting_values[5:]
            variable_revision_number = 4
        if variable_revision_number == 4:
            setting_values = setting_values[:4] + [False] + setting_values[4:]
            variable_revision_number = 5
        if variable_revision_number == 5:
            # added from_old_matlab
            setting_values = setting_values[:7] + [False] + setting_values[7:]
            variable_revision_number = 6
        if variable_revision_number == 6:
            # added go_to_website
            setting_values = setting_values[:8] + [False] + setting_values[8:]
            variable_revision_number = 7
        if variable_revision_number == 7:
            setting_values = setting_values[:8] + setting_values[9:]
            variable_revision_number = 8

        return setting_values, variable_revision_number
