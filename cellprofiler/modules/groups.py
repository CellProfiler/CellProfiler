import cellprofiler.icons
from cellprofiler.gui.help import PROTIP_RECOMEND_ICON, PROTIP_AVOID_ICON, TECH_NOTE_ICON, GROUPS_DISPLAY_TABLE

__doc__ = """
The <b>Groups</b> module organizes sets of images into groups.
<hr>
Once the images have been identified with the <b>Images</b> module, have had metadata associated with
them using the <b>Metadata</b> module, and have been assigned names by the <b>NamesAndTypes</b> module,
you have the option of further sub-dividing the image sets into groups that share a common feature.
Some downstream modules of CellProfiler are capable of processing groups of images in useful ways
(e.g., object tracking within a set of images comprising a time series, illumination correction within
a set of images comprising an experimental batch, data export for a set of images comprising a plate).

<h4>What is an image "group"?</h4>
<p>The key to understanding why grouping may be necessary is that CellProfiler processes the input images
sequentially and in the order given by the NamesAndTypes module. If you have multiple collections (or "groups")
of images that should be processed independently from each other, CellProfiler will simply finish processing
one collection and proceed to the next, ignoring any distinction between them unless told otherwise via
the <b>Groups</b> module.</p>

<p>To illustrate this idea, below are two examples where the grouping concept can be useful or important:
<ul>
<li>If you have time-lapse movie data that is in the form of individual image files, and you
are performing object tracking, it is important to indicate to CellProfiler that the end of a movie
indicates the end of a distinct data set. Without doing so, CellProfiler will simply take the first frame
of the next movie as a continuation of the previous one. If each set of
files that comprise a movie is defined using the <b>Metadata</b> module, the relevant metadata can
be used in this module to insure that object tracking only takes place within each movie.</li>
<li>If you are performing illumination correction for a screening experiment, we recommend
that the illumination function (an image which represents the overall background fluorescence)
be calculated on a per-plate basis. Since the illumination function is an aggregate of images from a
plate, running a pipeline must yield a single illumination function for each plate. Running this
pipeline multiple times, once for each plate, will give the desired result but would be tedious and
time-consuming. In this case, CellProfiler can use image grouping for this purpose; if plate metadata
can be defined by the <b>Metadata</b> module, grouping will enable you to
process images that have the same plate metadata together.</li>
</ul>
</p>

<h4>What are the inputs?</h4>
Using this module assumes that you have already adjusted the following Input modules:
<ul>
<li>Used the <b>Images</b> module to produce a list of images to analyze. </li>
<li>Used the <b>Metadata</b> module to produce metadata defining the distinct sub-divisions between groups of images.</li>
<li>Used the <b>NamesAndTypes</b> module to assign names to individual channels and create image sets.</li>
</ul>

<h4>What do the settings mean?</h4>
See below for help on the individual settings. Selecting this module will display a panel, allowing you to select
whether you want to create groups or not. A grouping may be defined as according to any or as many of the metadata
categories as defined by the <b>Metadata</b> module. By selecting a metadata tag from the drop-down for the
metadata category, the <b>Groups</b> module will sub-divide and assemble the image sets according to their unique
metadata value. Upon adding a metadata category, the two tables underneath will update to show the resultant
organization of the image sets for each group.</p>

<h4>What do I get as output?</h4>
The final product of the <b>Groups</b> module is a list defining subsets of image sets that will be processed
independently of the other subsets.
<ul>
<li>If no groups are defined, the Analysis modules in the rest of the pipeline will be applied to all images in
exactly the same way. </li>
<li>If groups are defined in the <b>Groups</b> module, then organizationally (and transparently to you),
CellProfiler will begin the analyses with the first image set of the group, end with the last image set of the
group, and then proceed to the next group. </li>
</ul>
<p>The two tables at the bottom provide the following information when a metadata category is selected:
<ul>
<li>The <i>grouping list</i>(top table) shows the unique values of the selected metadata under the "Group"
column; each of the unique values comprises a group. The "Count" column shows the number of image sets included
in a given group; this is useful as a "sanity check" to make sure that the expected numbers of images are present.</li>
<li>The <i>image set list</i> (bottom table) shows the file name and location of each of the image sets that
comprise the groups.</li>
</ul>
<table cellpadding="0" width="100%%">
<tr align="center"><td><img src="memory:%(GROUPS_DISPLAY_TABLE)s"></td></tr>
</table>
</p>

<h4>Available measurements</h4>
<ul>
<li><i>Group_Number:</i> The index of each grouping, as defined by the unique combinations of the metadata
tags specified. These are written to the per-image table.</li>
<li><i>Group_Index:</i> The index of each imaget set within each grouping, as defined by the <i>Group_Number</i>.
These are written to the per-image table.</li>
</ul>

<h4>Technical notes</h4>
To perform grouping, only one analysis worker (i.e., copy of CellProfiler) will be allocated to handle each group.
This means that you may have multiple workers created (as set under the Preferences), but only a subset of them
may actually be active, depending on the number of groups you have.
""" % globals()

import logging

logger = logging.getLogger(__name__)
import numpy as np
import os

import cellprofiler.module as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO
import cellprofiler.measurement as cpmeas


class Groups(cpm.Module):
    variable_revision_number = 2
    module_name = "Groups"
    category = "File Processing"

    IDX_GROUPING_METADATA_COUNT = 1

    def create_settings(self):
        self.pipeline = None
        self.metadata_keys = {}

        module_explanation = [
            "The %s module optionally allows you to split your list of images into image subsets" % self.module_name,
            "(groups) which will be processed independently of each other. Examples of",
            "groupings include screening batches, microtiter plates, time-lapse movies, etc."]
        self.set_notes([" ".join(module_explanation)])

        self.wants_groups = cps.Binary(
                "Do you want to group your images?", False, doc="""
            Select <i>%(YES)s</i> if you need to split your images into image subsets (or <i>groups</i>) such that each
            group is processed independently of each other. See the main module
            help for more details.""" % globals())

        self.grouping_text = cps.HTMLText(
                "", content="""
            Each unique metadata value (or combination of values)
            will be defined as a group""", size=(30, 2))
        self.grouping_metadata = []
        self.grouping_metadata_count = cps.HiddenCount(
                self.grouping_metadata,
                "grouping metadata count")
        self.add_grouping_metadata(can_remove=False)
        self.add_grouping_metadata_button = cps.DoSomething(
                "", "Add another metadata item", self.add_grouping_metadata)

        self.grouping_list = cps.Table("Grouping list", min_size=(300, 100), doc="""
            This list shows the unique values of the selected metadata under the "Group" column; each of the unique values
            comprises a group. The "Count" column shows the number of image sets that included in a given group; this
            is useful as a "sanity check", to make sure that the expected number of images are present. For example,
            if you are grouping by per-plate metadata from a 384-well assay with 2 sites per well consisting of 3 plates,
            you would expect to see 3 groups (each from the 3 unique plate IDs), with 384 wells &times; 2 sites/well
            = 768 image sets in each.""")

        self.image_set_list = cps.Table("Image sets", doc="""
            This list displays the file name and location of each of the image sets that comprise the
            group. For example, if you are grouping by per-plate metadata from a 384-well assay with 2 sites per well
            consisting of 3 plates, you would expect to see a table consisting of 3 plates &times; 384 wells/plate
            &times;2 sites/well = 2304 rows.""")

    def add_grouping_metadata(self, can_remove=True):
        group = cps.SettingsGroup()
        self.grouping_metadata.append(group)

        def get_group_metadata_choices(pipeline):
            choices = self.get_metadata_choices(pipeline, group)
            if len(choices) == 0:
                choices.append(cps.NONE)
            return choices

        if self.pipeline is not None:
            choices = get_group_metadata_choices(self.pipeline)
        else:
            choices = [cps.NONE]

        group.append("metadata_choice", cps.Choice(
                "Metadata category", choices,
                choices_fn=get_group_metadata_choices, doc="""
            Specify the metadata category with which to define a group. Once a selection
            is made, the two listings below will display the updated values:
            <ul>
            <li>The <i>grouping list</i> (top table) shows the unique values of the selected metadata under the
            "Group" column; each of the unique values comprises a group. The "Count" column shows the number of
            image sets included in a given group; this is useful as a "sanity check" to make sure that the
            expected numbers of images are present.</li>
            <li>The <i>image set list</i> (bottom table) shows the file name and location of each of the image
            sets that comprise the groups. In this example, the table has 26 rows, one for each of the DNA and
            GFP image sets defined by the <b>NamesAndTypes</b> module. </li>
            </ul>
            You may specify multiple metadata tags to group with by clicking the "Add" button. This would
            be necessary if a combination of metadata is required in order to define a group. Upon adding a
            metadata category, the two tables will update in the panels below showing the resulting organization
            of the image data for each group.

            <p>As an example, an time-lapse experiment consists of a set of movie images (indexed by a frame number), collected
            on a per-well basis. The plate, well, wavelength and frame number metadata have been extracted using the
            <b>Metadata</b> module. Using the <b>NamesAndTypes</b>
            module, the two image channels (OrigBlue, <i>w1</i> and OrigGreen, <i>w2</i>) have been
            set up in the following way:
            <table border="1" align="center">
            <tr><th><b>Image set number</b></th><th><b>OrigBlue (w1) file name</b></th><th><b>OrigGreen (w2) file name</b></th><th><b>Plate</b></th><th><b>Well</b></th><th><b>FrameNumber</b></th></tr>
            <tr><td>1</td><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w1</font>.tif</td><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w2</font>.tif</td><td>P-12345</td><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">t001</font></td></tr>
            <tr><td>2</td><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w1</font>.tif</td><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w2</font>.tif</td><td>P-12345</td><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">t002</font></td></tr>
            <tr><td>3</td><td>P-12345_<font color="#ce5f33">B01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w1</font>.tif</td><td>P-12345_<font color="#ce5f33">B01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w2</font>.tif</td><td>P-12345</td><td><font color="#ce5f33">B01</font></td><td><font color="#3dce33">t001</font></td></tr>
            <tr><td>4</td><td>P-12345_<font color="#ce5f33">B01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w1</font>.tif</td><td>P-12345_<font color="#ce5f33">B01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w2</font>.tif</td><td>P-12345</td><td><font color="#ce5f33">B01</font></td><td><font color="#3dce33">t002</font></td></tr>
            <tr><td>5</td><td>2-ABCDF_<font color="#ce5f33">A01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w1</font>.tif</td><td>2-ABCDF_<font color="#ce5f33">A01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w2</font>.tif</td><td>2-ABCDF_</td><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">t001</font></td></tr>
            <tr><td>6</td><td>2-ABCDF_<font color="#ce5f33">A01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w1</font>.tif</td><td>2-ABCDF_<font color="#ce5f33">A01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w2</font>.tif</td><td>2-ABCDF_</td><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">t002</font></td></tr>
            <tr><td>7</td><td>2-ABCDF_<font color="#ce5f33">B01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w1</font>.tif</td><td>2-ABCDF_<font color="#ce5f33">B01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w2</font>.tif</td><td>2-ABCDF_</td><td><font color="#ce5f33">B01</font></td><td><font color="#3dce33">t001</font></td></tr>
            <tr><td>8</td><td>2-ABCDF_<font color="#ce5f33">B01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w1</font>.tif</td><td>2-ABCDF_<font color="#ce5f33">B01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w2</font>.tif</td><td>2-ABCDF_</td><td><font color="#ce5f33">B01</font></td><td><font color="#3dce33">t002</font></td></tr>
            </table>
            We would like to perform object tracking for each movie, i.e., for each plate and well.
            Without the use of groups, even though image sets 1&ndash;2, 3&ndash;4, 5&ndash;6, and 7&ndash;8
            represent different movies, image set 3 will get processed immediately after image set 2,
            image set 5 after 4, and so on. For an object tracking assay, failure to recognize where the movies
            start and end would lead to incorrect tracking results.</p>

            <p>Selecting the <i>Plate</i> followed by the <i>Well</i> metadata as the metadata categories will
            create four groups based on the unique plate and well combinations:
            <table border="1" align="center">
            <tr><th colspan="2"><b>Grouping tags</b></th><th colspan="4"><b>Image set tags</b></th><th colspan="2"><b>Channels</b></th></tr>
            <tr><th><b>Group number</b></th><th><b>Group index</b></th><th><b>Image set number</b></th><th><b>Plate</b></th><th><b>Well</b></th><th><b>FrameNumber</b></th><th><b>OrigBlue</b></th><th><b>OrigGreen</b></th></tr>
            <tr><td rowspan="2">1</td><td>1</td><td>1</td><td>P-12345</td><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">t001</font></td><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w1</font>.tif</td><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w2</font>.tif</td></tr>
            <tr><td>2</td><td>2</td><td>P-12345</td><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">t002</font></td><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w1</font>.tif</td><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w2</font>.tif</td></tr>
            <tr><td rowspan="2">2</td><td>1</td><td>3</td><td>P-12345</td><td><font color="#ce5f33">B01</font></td><td><font color="#3dce33">t001</font></td><td>P-12345_<font color="#ce5f33">B01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w1</font>.tif</td><td>P-12345_<font color="#ce5f33">B01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w2</font>.tif</td></tr>
            <tr><td>2</td><td>4</td><td>P-12345</td><td><font color="#ce5f33">B01</font></td><td><font color="#3dce33">t002</font></td><td>P-12345_<font color="#ce5f33">B01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w1</font>.tif</td><td>P-12345_<font color="#ce5f33">B01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w2</font>.tif</td></tr>
            <tr><td rowspan="2">3</td><td>1</td><td>5</td><td>2-ABCDF</td><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">t001</font></td><td>2-ABCDF_<font color="#ce5f33">A01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w1</font>.tif</td><td>2-ABCDF_<font color="#ce5f33">A01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w2</font>.tif</td></tr>
            <tr><td>2</td><td>6</td><td>2-ABCDF</td><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">t002</font></td><td>2-ABCDF_<font color="#ce5f33">A01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w1</font>.tif</td><td>2-ABCDF_<font color="#ce5f33">A01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w2</font>.tif</td></tr>
            <tr><td rowspan="2">4</td><td>1</td><td>7</td><td>2-ABCDF</td><td><font color="#ce5f33">B01</font></td><td><font color="#3dce33">t001</font></td><td>2-ABCDF_<font color="#ce5f33">B01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w1</font>.tif</td><td>2-ABCDF_<font color="#ce5f33">B01</font>_<font color="#3dce33">t001</font>_<font color="#33bbce">w2</font>.tif</td></tr>
            <tr><td>2</td><td>8</td><td>2-ABCDF</td><td><font color="#ce5f33">B01</font></td><td><font color="#3dce33">t002</font></td><td>2-ABCDF_<font color="#ce5f33">B01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w1</font>.tif</td><td>2-ABCDF_<font color="#ce5f33">B01</font>_<font color="#3dce33">t002</font>_<font color="#33bbce">w2</font>.tif</td></tr>
            </table>
            Each group will be processed independently from the others, which is the desired behavior.
            </p>"""))

        group.append("divider", cps.Divider())
        group.can_remove = can_remove
        if can_remove:
            group.append("remover", cps.RemoveSettingButton(
                    "", "Remove this metadata item",
                    self.grouping_metadata, group))

    def get_metadata_choices(self, pipeline, group):
        if self.pipeline is not None:
            return sorted(self.metadata_keys)
        #
        # Unfortunate - an expensive operation to find the possible metadata
        #               keys from one of the columns in an image set.
        # Just fake it into having something that will work
        #
        return [group.metadata_choice.value]

    def settings(self):
        result = [self.wants_groups, self.grouping_metadata_count]
        for group in self.grouping_metadata:
            result += [group.metadata_choice]
        return result

    def visible_settings(self):
        result = [self.wants_groups]
        if self.wants_groups:
            for group in self.grouping_metadata:
                result += [group.metadata_choice]
                if group.can_remove:
                    result += [group.remover]
                result += [group.divider]
            result += [self.add_grouping_metadata_button,
                       self.grouping_list, self.image_set_list]
        return result

    def help_settings(self):
        result = [self.wants_groups,
                  self.grouping_metadata[0].metadata_choice,
                  self.grouping_list, self.image_set_list]
        return result

    def prepare_settings(self, setting_values):
        nmetadata = int(setting_values[self.IDX_GROUPING_METADATA_COUNT])
        while len(self.grouping_metadata) > nmetadata:
            del self.grouping_metadata[-1]

        while len(self.grouping_metadata) < nmetadata:
            self.add_grouping_metadata()

    def on_activated(self, workspace):
        self.pipeline = workspace.pipeline
        self.workspace = workspace
        assert isinstance(self.pipeline, cpp.Pipeline)
        if self.wants_groups:
            self.metadata_keys = []
            self.image_sets_initialized = workspace.refresh_image_set()
            self.metadata_keys = list(
                    self.pipeline.get_available_metadata_keys().keys())
            is_valid = True
            for group in self.grouping_metadata:
                try:
                    group.metadata_choice.test_valid(self.pipeline)
                except:
                    is_valid = False
            if is_valid:
                self.update_tables()
        else:
            self.image_sets_initialized = False

    def on_deactivated(self):
        self.pipeline = None

    def on_setting_changed(self, setting, pipeline):
        if (setting == self.wants_groups and self.wants_groups and
                not self.image_sets_initialized):
            workspace = self.workspace
            self.on_deactivated()
            self.on_activated(workspace)
            needs_prepare_run = False
        else:
            needs_prepare_run = True
        #
        # Unfortunately, test_valid has the side effect of getting
        # the choices set which is why it's called here
        #
        is_valid = True
        for group in self.grouping_metadata:
            try:
                group.metadata_choice.test_valid(pipeline)
            except:
                is_valid = False
        if is_valid:
            if needs_prepare_run:
                result = self.prepare_run(self.workspace)
                if not result:
                    return
            self.update_tables()

    def update_tables(self):
        if self.wants_groups:
            try:
                if not self.workspace.refresh_image_set():
                    return
            except:
                return
            m = self.workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            channel_descriptors = m.get_channel_descriptors()

            self.grouping_list.clear_columns()
            self.grouping_list.clear_rows()
            self.image_set_list.clear_columns()
            self.image_set_list.clear_rows()
            metadata_key_names = [
                group.metadata_choice.value
                for group in self.grouping_metadata
                if group.metadata_choice.value != "None"]
            metadata_feature_names = ["_".join((cpmeas.C_METADATA, key))
                                      for key in metadata_key_names]
            metadata_key_names = [
                x[(len(cpmeas.C_METADATA) + 1):]
                for x in metadata_feature_names]
            image_set_feature_names = [
                                          cpmeas.GROUP_NUMBER, cpmeas.GROUP_INDEX] + metadata_feature_names
            self.image_set_list.insert_column(0, "Group number")
            self.image_set_list.insert_column(1, "Group index")

            for i, key in enumerate(metadata_key_names):
                for l, offset in ((self.grouping_list, 0),
                                  (self.image_set_list, 2)):
                    l.insert_column(i + offset, "Group: %s" % key)

            self.grouping_list.insert_column(len(metadata_key_names), "Count")

            image_numbers = m.get_image_numbers()
            group_indexes = m[cpmeas.IMAGE,
                              cpmeas.GROUP_INDEX,
                              image_numbers][:]
            group_numbers = m[cpmeas.IMAGE,
                              cpmeas.GROUP_NUMBER,
                              image_numbers][:]
            counts = np.bincount(group_numbers)
            first_indexes = np.argwhere(group_indexes == 1).flatten()
            group_keys = [
                m[cpmeas.IMAGE, feature, image_numbers]
                for feature in metadata_feature_names]
            k_count = sorted([(group_numbers[i],
                               [x[i] for x in group_keys],
                               counts[group_numbers[i]])
                              for i in first_indexes])
            for group_number, group_key_values, c in k_count:
                row = group_key_values + [c]
                self.grouping_list.data.append(row)

            for i, iscd in enumerate(channel_descriptors):
                assert isinstance(iscd, cpp.Pipeline.ImageSetChannelDescriptor)
                image_name = iscd.name
                idx = len(image_set_feature_names)
                self.image_set_list.insert_column(idx, "Path: %s" % image_name)
                self.image_set_list.insert_column(idx + 1, "File: %s" % image_name)
                if iscd.channel_type == iscd.CT_OBJECTS:
                    image_set_feature_names.append(
                            cpmeas.C_OBJECTS_PATH_NAME + "_" + iscd.name)
                    image_set_feature_names.append(
                            cpmeas.C_OBJECTS_FILE_NAME + "_" + iscd.name)
                else:
                    image_set_feature_names.append(
                            cpmeas.C_PATH_NAME + "_" + iscd.name)
                    image_set_feature_names.append(
                            cpmeas.C_FILE_NAME + "_" + iscd.name)

            all_features = [m[cpmeas.IMAGE, ftr, image_numbers]
                            for ftr in image_set_feature_names]
            order = np.lexsort((group_indexes, group_numbers))

            for idx in order:
                row = [unicode(x[idx]) for x in all_features]
                self.image_set_list.data.append(row)

    def get_groupings(self, workspace):
        '''Return the image groupings of the image sets in an image set list

        returns a tuple of key_names and group_list:
        key_names - the names of the keys that identify the groupings
        group_list - a sequence composed of two-tuples.
                     the first element of the tuple has the values for
                     the key_names for this group.
                     the second element of the tuple is a sequence of
                     image numbers comprising the image sets of the group
        For instance, an experiment might have key_names of 'Metadata_Row'
        and 'Metadata_Column' and a group_list of:
        [ ({'Row':'A','Column':'01'), [0,96,192]),
          (('Row':'A','Column':'02'), [1,97,193]),... ]
        '''
        if not self.wants_groups:
            return
        key_list = self.get_grouping_tags()
        m = workspace.measurements
        for key in key_list:
            if key not in m.get_feature_names(cpmeas.IMAGE):
                if key.startswith(cpmeas.C_METADATA):
                    key = key[len(cpmeas.C_METADATA) + 1:]
                workspace.pipeline.report_prepare_run_error(
                        self,
                        ('The groups module is misconfigured. "%s" was chosen as\n'
                         'one of the metadata tags, but that metadata tag is not\n'
                         'defined in the Metadata module.') % key)
                return None
        return key_list, m.get_groupings(key_list)

    def get_grouping_tags(self):
        '''Return the metadata keys used for grouping'''
        if not self.wants_groups:
            return None
        return ["_".join((cpmeas.C_METADATA, g.metadata_choice.value))
                for g in self.grouping_metadata]

    def change_causes_prepare_run(self, setting):
        '''Return True if changing the setting passed changes the image sets

        setting - the setting that was changed
        '''
        return setting in self.settings()

    def is_load_module(self):
        '''Marks this module as a module that affects the image sets

        Groups is a load module because it can reorder image sets, but only
        if grouping is turned on.
        '''
        return self.wants_groups.value

    @classmethod
    def is_input_module(self):
        return True

    def prepare_run(self, workspace):
        '''Reorder the image sets and assign group number and index'''
        if workspace.pipeline.in_batch_mode():
            return True

        if not self.wants_groups:
            return True

        if len(workspace.measurements.get_image_numbers()) == 0:
            return False

        result = self.get_groupings(workspace)
        if result is None:
            return False
        key_list, groupings = result
        #
        # Sort the groupings by key
        #
        groupings = sorted(groupings)
        #
        # Create arrays of group number, group_index and image_number
        #
        group_numbers = np.hstack([
                                      np.ones(len(image_numbers), int) * (i + 1)
                                      for i, (keys, image_numbers) in enumerate(groupings)])
        group_indexes = np.hstack([
                                      np.arange(len(image_numbers)) + 1
                                      for keys, image_numbers in groupings])
        image_numbers = np.hstack([
                                      image_numbers for keys, image_numbers in groupings])
        order = np.lexsort((group_indexes, group_numbers))
        group_numbers = group_numbers[order]
        group_indexes = group_indexes[order]

        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        #
        # Downstream processing requires that image sets be ordered by
        # increasing group number, then increasing group index.
        #
        new_image_numbers = np.zeros(np.max(image_numbers) + 1, int)
        new_image_numbers[image_numbers[order]] = np.arange(len(image_numbers)) + 1
        m.reorder_image_measurements(new_image_numbers)
        m.add_all_measurements(cpmeas.IMAGE, cpmeas.GROUP_NUMBER, group_numbers)
        m.add_all_measurements(cpmeas.IMAGE, cpmeas.GROUP_INDEX, group_indexes)
        m.set_grouping_tags(self.get_grouping_tags())
        return True

    def run(self, workspace):
        pass

    def get_measurement_columns(self, pipeline):
        '''Return the measurments recorded by this module

        GroupNumber and GroupIndex are accounted for by the pipeline itself.
        '''
        result = []
        if self.wants_groups:
            result.append((cpmeas.EXPERIMENT,
                           cpmeas.M_GROUPING_TAGS,
                           cpmeas.COLTYPE_VARCHAR))
            #
            # These are bound to be produced elsewhere, but it is quite
            # computationally expensive to find that out. If they are
            # duplicated by another module, no big deal.
            #
            for ftr in self.get_grouping_tags():
                result.append((cpmeas.IMAGE, ftr, cpmeas.COLTYPE_VARCHAR))
        return result

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if variable_revision_number == 1:
            #
            # Remove the image name from the settings
            #
            new_setting_values = \
                setting_values[:(self.IDX_GROUPING_METADATA_COUNT + 1)]
            for i in range(int(setting_values[self.IDX_GROUPING_METADATA_COUNT])):
                new_setting_values.append(
                        setting_values[self.IDX_GROUPING_METADATA_COUNT + 2 + i * 2])
            setting_values = new_setting_values
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

    def volumetric(self):
        return True
