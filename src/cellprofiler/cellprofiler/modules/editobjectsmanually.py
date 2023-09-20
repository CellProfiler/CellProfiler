from cellprofiler_core.constants.measurement import COLTYPE_INTEGER
from cellprofiler_core.constants.measurement import FF_CHILDREN_COUNT
from cellprofiler_core.constants.measurement import FF_PARENT
from cellprofiler_core.image import ObjectsImage
from cellprofiler_core.module import Identify
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import LabelName
from cellprofiler_core.utilities.core.module.identify import (
    add_object_count_measurements,
)
from cellprofiler_core.utilities.core.module.identify import (
    add_object_location_measurements_ijv,
)
from cellprofiler_core.utilities.core.module.identify import (
    get_object_measurement_columns,
)

from cellprofiler.modules import _help

__doc__ = """\
EditObjectsManually
===================

**EditObjectsManually** allows you create, remove and edit objects
previously defined.

The interface will show the image that you selected as the guiding
image, overlaid with colored outlines of the selected objects (or filled
objects if you choose). This module allows you to remove or edit
specific objects by pointing and clicking to select objects for removal
or editing. Once editing is complete, the module displays the objects as
originally identified (left) and the objects that remain after this
module (right). More detailed Help is provided in the editing window via
the ‘?’ button. The pipeline pauses once per processed image when it
reaches this module. You must press the *Done* button to accept the
selected objects and continue the pipeline.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Image measurements:**

-  *Count:* The number of edited objects in the image.

**Object measurements:**

-  *Location\_X, Location\_Y:* The pixel (X,Y) coordinates of the center
   of mass of the edited objects.

See also
^^^^^^^^

See also **FilterObjects**, **MaskObject**, **OverlayOutlines**,
**ConvertToImage**.

{HELP_ON_SAVING_OBJECTS}

""".format(
    **{"HELP_ON_SAVING_OBJECTS": _help.HELP_ON_SAVING_OBJECTS}
)

import os
import numpy

from cellprofiler_core.object import Objects
from cellprofiler_core.setting import Binary

from cellprofiler_core.utilities.pathname import pathname2url

###########################################
#
# Choices for the "do you want to renumber your objects" setting
#
###########################################
R_RENUMBER = "Renumber"
R_RETAIN = "Retain"


class EditObjectsManually(Identify):
    category = "Object Processing"
    variable_revision_number = 4
    module_name = "EditObjectsManually"

    def create_settings(self):
        """Create your settings by subclassing this function

        create_settings is called at the end of initialization.

        You should create the setting variables for your module here:
            # Ask the user for the input image
            self.image_name = .ImageSubscriber(...)
            # Ask the user for the name of the output image
            self.output_image = .ImageName(...)
            # Ask the user for a parameter
            self.smoothing_size = .Float(...)
        """
        self.object_name = LabelSubscriber(
            "Select the objects to be edited",
            "None",
            doc="""\
Choose a set of previously identified objects
for editing, such as those produced by one of the
**Identify** modules (e.g., "*IdentifyPrimaryObjects*", "*IdentifySecondaryObjects*" etc.).""",
        )

        self.filtered_objects = LabelName(
            "Name the edited objects",
            "EditedObjects",
            doc="""\
Enter the name for the objects that remain
after editing. These objects will be available for use by
subsequent modules.""",
        )

        self.allow_overlap = Binary(
            "Allow overlapping objects?",
            False,
            doc="""\
**EditObjectsManually** can allow you to edit an object so that it
overlaps another or it can prevent you from overlapping one object with
another. Objects such as worms or the neurites of neurons may cross each
other and might need to be edited with overlapping allowed, whereas a
monolayer of cells might be best edited with overlapping off.
Select "*Yes*" to allow overlaps or select "*No*" to prevent them.
"""
            % globals(),
        )

        self.renumber_choice = Choice(
            "Numbering of the edited objects",
            [R_RENUMBER, R_RETAIN],
            doc="""\
Choose how to number the objects that remain after editing, which
controls how edited objects are associated with their predecessors:

-  *%(R_RENUMBER)s:* The module will number the objects that remain
   using consecutive numbers. This is a good choice if you do not plan
   to use measurements from the original objects and you only want to
   use the edited objects in downstream modules; the objects that remain
   after editing will not have gaps in numbering where removed objects
   are missing.
-  *%(R_RETAIN)s:* This option will retain each object’s original
   number so that the edited object’s number matches its original
   number. This allows any measurements you make from the edited objects
   to be directly aligned with measurements you might have made of the
   original, unedited objects (or objects directly associated with
   them).
"""
            % globals(),
        )

        self.wants_image_display = Binary(
            "Display a guiding image?",
            True,
            doc="""\
Select "*Yes*" to display an image and outlines of the objects.

Select "*No*" if you do not want a guide image while editing.
"""
            % globals(),
        )

        self.image_name = ImageSubscriber(
            "Select the guiding image",
            "None",
            doc="""\
*(Used only if a guiding image is desired)*

This is the image that will appear when editing objects. Choose an image
supplied by a previous module.
""",
        )

    def settings(self):
        """Return the settings to be loaded or saved to/from the pipeline

        These are the settings (from cellprofiler_core.settings) that are
        either read from the strings in the pipeline or written out
        to the pipeline. The settings should appear in a consistent
        order so they can be matched to the strings in the pipeline.
        """
        return [
            self.object_name,
            self.filtered_objects,
            self.renumber_choice,
            self.wants_image_display,
            self.image_name,
            self.allow_overlap,
        ]

    def visible_settings(self):
        result = [
            self.object_name,
            self.filtered_objects,
            self.allow_overlap,
            self.renumber_choice,
            self.wants_image_display,
        ]

        if self.wants_image_display:
            result += [self.image_name]
        return result

    def run(self, workspace):
        """Run the module

        workspace    - The workspace contains
            pipeline     - instance of cpp for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - the parent frame to whatever frame is created. None means don't draw.
        """
        orig_objects_name = self.object_name.value
        filtered_objects_name = self.filtered_objects.value

        orig_objects = workspace.object_set.get_objects(orig_objects_name)
        assert isinstance(orig_objects, Objects)
        orig_labels = [l for l, c in orig_objects.get_labels()]

        if self.wants_image_display:
            guide_image = workspace.image_set.get_image(self.image_name.value)
            guide_image = guide_image.pixel_data
            if guide_image.dtype == bool:
                guide_image = guide_image.astype(int)
            if numpy.any(guide_image != numpy.min(guide_image)):
                guide_image = (guide_image - numpy.min(guide_image)) / (
                    numpy.max(guide_image) - numpy.min(guide_image)
                )
        else:
            guide_image = None
        filtered_labels = workspace.interaction_request(
            self, orig_labels, guide_image, workspace.measurements.image_set_number
        )
        if filtered_labels is None:
            # Ask whoever is listening to stop doing stuff
            workspace.cancel_request()
            # Have to soldier on until the cancel takes effect...
            filtered_labels = orig_labels
        #
        # Renumber objects consecutively if asked to do so
        #
        unique_labels = numpy.unique(numpy.array(filtered_labels))
        unique_labels = unique_labels[unique_labels != 0]
        object_count = len(unique_labels)
        if self.renumber_choice == R_RENUMBER:
            mapping = numpy.zeros(
                1 if len(unique_labels) == 0 else numpy.max(unique_labels) + 1, int
            )
            mapping[unique_labels] = numpy.arange(1, object_count + 1)
            filtered_labels = [mapping[l] for l in filtered_labels]
        #
        # Make the objects out of the labels
        #
        filtered_objects = Objects()
        i, j = numpy.mgrid[
            0 : filtered_labels[0].shape[0], 0 : filtered_labels[0].shape[1]
        ]
        ijv = numpy.zeros((0, 3), filtered_labels[0].dtype)
        for l in filtered_labels:
            ijv = numpy.vstack(
                (ijv, numpy.column_stack((i[l != 0], j[l != 0], l[l != 0])))
            )
        filtered_objects.set_ijv(ijv, orig_labels[0].shape)
        if orig_objects.has_unedited_segmented():
            filtered_objects.unedited_segmented = orig_objects.unedited_segmented
        if orig_objects.parent_image is not None:
            filtered_objects.parent_image = orig_objects.parent_image
        workspace.object_set.add_objects(filtered_objects, filtered_objects_name)
        #
        # Add parent/child & other measurements
        #
        m = workspace.measurements
        child_count, parents = orig_objects.relate_children(filtered_objects)
        m.add_measurement(
            filtered_objects_name, FF_PARENT % orig_objects_name, parents,
        )
        m.add_measurement(
            orig_objects_name, FF_CHILDREN_COUNT % filtered_objects_name, child_count,
        )
        #
        # The object count
        #
        add_object_count_measurements(m, filtered_objects_name, object_count)
        #
        # The object locations
        #
        add_object_location_measurements_ijv(m, filtered_objects_name, ijv)

        workspace.display_data.orig_ijv = orig_objects.ijv
        workspace.display_data.filtered_ijv = filtered_objects.ijv
        workspace.display_data.shape = orig_labels[0].shape

    def display(self, workspace, figure):
        orig_ijv = workspace.display_data.orig_ijv
        filtered_ijv = workspace.display_data.filtered_ijv
        shape = workspace.display_data.shape
        figure.set_subplots((2, 1))
        ax0 = figure.subplot_imshow_ijv(
            0, 0, orig_ijv, shape=shape, title=self.object_name.value
        )
        figure.subplot_imshow_ijv(
            1,
            0,
            filtered_ijv,
            shape=shape,
            title=self.filtered_objects.value,
            sharex=ax0,
            sharey=ax0,
        )

    def run_as_data_tool(self):
        from cellprofiler.gui.editobjectsdlg import EditObjectsDialog
        import wx
        from wx.lib.filebrowsebutton import FileBrowseButton
        from cellprofiler_core.reader import get_image_reader
        import imageio

        with wx.Dialog(None) as dlg:
            dlg.Title = "Choose files for editing"
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            dlg.Sizer.Add(sub_sizer, 0, wx.EXPAND | wx.ALL, 5)
            new_or_existing_rb = wx.RadioBox(
                dlg, style=wx.RA_VERTICAL, choices=("New", "Existing")
            )
            sub_sizer.Add(new_or_existing_rb, 0, wx.EXPAND)
            objects_file_fbb = FileBrowseButton(
                dlg,
                size=(300, -1),
                fileMask="Objects file (*.tif, *.tiff, *.png, *.bmp, *.jpg)|*.tif;*.tiff;*.png;*.bmp;*.jpg",
                dialogTitle="Select objects file",
                labelText="Objects file:",
            )
            objects_file_fbb.Enable(False)
            sub_sizer.AddSpacer(5)
            sub_sizer.Add(objects_file_fbb, 0, wx.ALIGN_TOP | wx.ALIGN_RIGHT)

            def on_radiobox(event):
                objects_file_fbb.Enable(new_or_existing_rb.GetSelection() == 1)

            new_or_existing_rb.Bind(wx.EVT_RADIOBOX, on_radiobox)

            image_file_fbb = FileBrowseButton(
                dlg,
                size=(300, -1),
                fileMask="Objects file (*.tif, *.tiff, *.png, *.bmp, *.jpg)|*.tif;*.tiff;*.png;*.bmp;*.jpg",
                dialogTitle="Select guide image file",
                labelText="Guide image:",
            )
            dlg.Sizer.Add(image_file_fbb, 0, wx.EXPAND | wx.ALL, 5)

            allow_overlap_checkbox = wx.CheckBox(dlg, -1, "Allow objects to overlap")
            allow_overlap_checkbox.Value = True
            dlg.Sizer.Add(allow_overlap_checkbox, 0, wx.EXPAND | wx.ALL, 5)

            buttons = wx.StdDialogButtonSizer()
            dlg.Sizer.Add(
                buttons, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT | wx.ALL, 5
            )
            buttons.Add(wx.Button(dlg, wx.ID_OK))
            buttons.Add(wx.Button(dlg, wx.ID_CANCEL))
            buttons.Realize()
            dlg.Fit()
            result = dlg.ShowModal()
            if result != wx.ID_OK:
                return
            self.allow_overlap.value = allow_overlap_checkbox.Value
            fullname = objects_file_fbb.GetValue()
            guidename = image_file_fbb.GetValue()

        if new_or_existing_rb.GetSelection() == 1:
            provider = ObjectsImage("InputObjects", pathname2url(fullname), None, None)
            image = provider.provide_image(None)
            pixel_data = image.pixel_data
            labels = [pixel_data[:, :, i] for i in range(pixel_data.shape[2])]
        else:
            labels = None
        #
        # Load the guide image
        #
        guide_image_reader = get_image_reader(guidename)
        guide_image = guide_image_reader.read()
        if numpy.min(guide_image) != numpy.max(guide_image):
            guide_image = (guide_image - numpy.min(guide_image)) / (
                numpy.max(guide_image) - numpy.min(guide_image)
            )
        if labels is None:
            shape = guide_image.shape[:2]
            labels = [numpy.zeros(shape, int)]
        with EditObjectsDialog(
            guide_image, labels, self.allow_overlap, self.object_name.value
        ) as dialog_box:
            result = dialog_box.ShowModal()
            if result != wx.OK:
                return
            labels = dialog_box.labels
        with wx.FileDialog(None, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:

            dlg.Path = fullname
            dlg.Wildcard = (
                "Object image file (*.tif,*.tiff)|*.tif;*.tiff|"
                "Ilastik project file (*.ilp)|*.ilp"
            )
            result = dlg.ShowModal()
            fullname = dlg.Path
            if result == wx.ID_OK:
                if fullname.endswith(".ilp"):
                    self.save_into_ilp(fullname, labels, guidename)
                else:
                    if os.path.exists(fullname):
                        os.unlink(fullname)
                    imageio.volwrite(fullname, numpy.stack(labels, axis=-1))

    def save_into_ilp(self, project_name, labels, guidename):
        import h5py
        import wx

        with h5py.File(project_name) as f:
            g = f["DataSets"]
            for k in g:
                data_item = g[k]
                if data_item.attrs.get("fileName") == guidename:
                    break
            else:
                wx.MessageBox(
                    "Sorry, could not find the file, %s, in the project, %s"
                    % (guidename, project_name)
                )
            project_labels = data_item["labels"]["data"]
            mask = numpy.ones(project_labels.shape[2:4], project_labels.dtype)
            for label in labels:
                mask[label != 0] = 2
            #
            # "only" use the first 100,000 points in the image
            #
            subsample = 100000
            npts = numpy.prod(mask.shape)
            if npts > subsample:
                r = numpy.random.RandomState()
                r.seed(numpy.sum(mask) % (2 ** 16))
                i, j = numpy.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
                i0 = i[mask == 1]
                j0 = j[mask == 1]
                i1 = i[mask == 2]
                j1 = j[mask == 2]
                if len(i1) < subsample / 2:
                    p0 = r.permutation(len(i0))[: (subsample - len(i1))]
                    p1 = numpy.arange(len(i1))
                elif len(i0) < subsample / 2:
                    p0 = numpy.arange(len(i0))
                    p1 = r.permutation(len(i1))[: (subsample - len(i0))]
                else:
                    p0 = r.permutation(len(i0))[: (subsample / 2)]
                    p1 = r.permutation(len(i1))[: (subsample / 2)]
                mask_copy = numpy.zeros(mask.shape, mask.dtype)
                mask_copy[i0[p0], j0[p0]] = 1
                mask_copy[i1[p1], j1[p1]] = 2
                if "prediction" in data_item:
                    prediction = data_item["prediction"]
                    if numpy.max(prediction[0, 0, :, :, 0]) > 0.5:
                        # Only do if prediction was done (otherwise all == 0)
                        for n in range(2):
                            p = prediction[0, 0, :, :, n]
                            bad = (p < 0.5) & (mask == n + 1)
                            mask_copy[i[bad], j[bad]] = n + 1
                mask = mask_copy
            project_labels[0, 0, :, :, 0] = mask

    def handle_interaction(self, orig_labels, guide_image, image_set_number):
        from cellprofiler.gui.editobjectsdlg import EditObjectsDialog
        from wx import OK

        title = "%s #%d, image cycle #%d: " % (
            self.module_name,
            self.module_num,
            image_set_number,
        )
        title += (
            "Create, remove and edit %s. Click Help for full instructions"
            % self.object_name.value
        )
        with EditObjectsDialog(
            guide_image, orig_labels, self.allow_overlap, title
        ) as dialog_box:
            result = dialog_box.ShowModal()
            if result != OK:
                return None
            return dialog_box.labels

    def get_measurement_columns(self, pipeline):
        """Return information to use when creating database columns"""
        orig_image_name = self.object_name.value
        filtered_image_name = self.filtered_objects.value
        columns = get_object_measurement_columns(filtered_image_name)
        columns += [
            (
                orig_image_name,
                FF_CHILDREN_COUNT % filtered_image_name,
                COLTYPE_INTEGER,
            ),
            (filtered_image_name, FF_PARENT % orig_image_name, COLTYPE_INTEGER,),
        ]
        return columns

    def get_object_dictionary(self):
        """Return the dictionary that's used by identify.get_object_*"""
        return {self.filtered_objects.value: [self.object_name.value]}

    def get_categories(self, pipeline, object_name):
        """Get the measurement categories produced by this module

        pipeline - pipeline being run
        object_name - fetch categories for this object
        """
        categories = self.get_object_categories(
            pipeline, object_name, self.get_object_dictionary()
        )
        return categories

    def get_measurements(self, pipeline, object_name, category):
        """Get the measurement features produced by this module

        pipeline - pipeline being run
        object_name - fetch features for this object
        category - fetch features for this category
        """
        measurements = self.get_object_measurements(
            pipeline, object_name, category, self.get_object_dictionary()
        )
        return measurements

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            # Added wants image + image
            setting_values = setting_values + ["No", "None"]
            variable_revision_number = 2

        if variable_revision_number == 2:
            # Added allow overlap, default = False
            setting_values = setting_values + ["No"]
            variable_revision_number = 3

        if variable_revision_number == 3:
            # Remove wants_outlines, outlines_name
            setting_values = setting_values[:2] + setting_values[4:]
            variable_revision_number = 4

        return setting_values, variable_revision_number
