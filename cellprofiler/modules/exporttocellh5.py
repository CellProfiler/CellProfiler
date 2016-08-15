'''<b>ExportToCellH5</b> exports measurements, objects and object relationships,
and images to the CellH5 data format.
<hr>
<h4>File structure</h4>
In multiprocessing-mode, CellProfiler will create satellite .cellh5 files that
are linked to the one that you specify using this module. The only thing
to note is that you must keep all .cellh5 files that are generated together
if you move them to a new folder.
'''

import os
import tempfile

import h5py
import numpy as np
import scipy.ndimage

import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.setting as cps
from cellprofiler.gui.help import \
    USING_METADATA_TAGS_REF, USING_METADATA_HELP_REF
from cellprofiler.modules.identify import R_PARENT
from cellprofiler.preferences import \
    IO_FOLDER_CHOICE_HELP_TEXT, IO_WITH_METADATA_HELP_TEXT
from cellprofiler.setting import YES, NO
from cellprofiler.gui.help import \
    USING_METADATA_TAGS_REF, USING_METADATA_HELP_REF
import cellh5
import cellh5.cellh5write
import numpy as np

OFF_OBJECTS_COUNT = 0
OFF_IMAGES_COUNT = 1

COLORS = [("Red", "0xFF0000"),
          ("Green", "0x00FF00"),
          ("Blue", "0x0000FF")]


class ExportToCellH5(cpm.Module):
    #
    # TODO: model z and t. Currently, CellProfiler would analyze each
    #       stack plane independently. I think the easiest way to handle
    #       z and t would be to add them to the site path if they are
    #       used in the experiment (e.g. a time series would have a
    #       path of "/plate/well/site/time")
    #
    #       I can add two more optional metadata keys that would let
    #       users capture this.
    #
    #       The more-complicated choice would be to store the data in a
    #       stack which would mean reworking the indices in every segmentation
    #       after the first. There are some legacy measurements that are
    #       really object numbers, so these would require a lot of work
    #       to get right. Also, the resulting segmentations are a little
    #       artificial since they seem to say that every object is one
    #       pixel thick in the T or Z direction.
    #

    module_name = "ExportToCellH5"
    variable_revision_number = 1
    category = ["File Processing"]

    SUBFILE_KEY = "subfile"
    IGNORE_METADATA = "None"

    def create_settings(self):
        '''Create the settings for the ExportToCellH5 module'''
        self.directory = cps.DirectoryPath(
                "Output file location",
                doc="""
            This setting lets you choose the folder for the output files.
            %(IO_FOLDER_CHOICE_HELP_TEXT)s
            """ % globals())

        def get_directory_fn():
            '''Get the directory for the CellH5 file'''
            return self.directory.get_absolute_path()

        def set_directory_fn(path):
            dir_choice, custom_path = self.directory.get_parts_from_path(path)
            self.directory.join_parts(dir_choice, custom_path)

        self.file_name = cps.FilenameText(
                "Output file name", "DefaultOut.ch5",
                get_directory_fn=get_directory_fn,
                set_directory_fn=set_directory_fn,
                metadata=True,
                browse_msg="Choose CellH5 file",
                mode=cps.FilenameText.MODE_APPEND,
                exts=[("CellH5 file (*.cellh5)", "*.ch5"),
                      ("HDF5 file (*.h5)", "*.h5"),
                      ("All files (*.*", "*.*")],
                doc="""
            This setting lets you name your CellH5 file. If you choose an
            existing file, CellProfiler will add new data to the file
            or overwrite existing locations.
            <p>%(IO_WITH_METADATA_HELP_TEXT)s %(USING_METADATA_TAGS_REF)s.
            For instance, if you have a metadata tag named
            "Plate", you can create a per-plate folder by selecting one the subfolder options
            and then specifying the subfolder name as "\g&lt;Plate&gt;". The module will
            substitute the metadata values for the current image set for any metadata tags in the
            folder name.%(USING_METADATA_HELP_REF)s.</p>

            """ % globals())
        self.overwrite_ok = cps.Binary(
                "Overwrite existing data without warning?", False,
                doc="""
            Select <i>%(YES)s</i> to automatically overwrite any existing data
            for a site. Select <i>%(NO)s</i> to be prompted first.

            If you are running the pipeline on a computing cluster,
            select <i>%(YES)s</i> unless you want execution to stop because you
            will not be prompted to intervene. Also note that two instances
            of CellProfiler cannot write to the same file at the same time,
            so you must ensure that separate names are used on a cluster.
            """ % globals())
        self.repack = cps.Binary(
                "Repack after analysis", True,
                doc="""
            This setting determines whether CellProfiler in multiprocessing mode
            repacks the data at the end of analysis. If you select <i>%(YES)s</i>,
            CellProfiler will combine all of the satellite files into a single
            file upon completion. This option requires some extra temporary disk
            space and takes some time at the end of analysis, but results in
            a single file which may occupy less disk space. If you select
            <i>%(NO)s</i>, CellProfiler will create a master file using the
            name that you give and this file will have links to individual
            data files that contain the actual data. Using the data generated by
            this option requires that you keep the master file and the linked
            files together when copying them to a new folder.
            """ % globals())
        self.plate_metadata = cps.Choice(
                "Plate metadata", [], value="Plate",
                choices_fn=self.get_metadata_choices,
                doc="""
            This is the metadata tag that identifies the plate name of
            the images for the current cycle. Choose <i>None</i> if
            your assay does not have metadata for plate name. If your
            assay is slide-based, you can use a metadata item that identifies
            the slide as the choice for this setting and set the well
            and site metadata items to <i>None</i>.""")
        self.well_metadata = cps.Choice(
                "Well metadata", [], value="Well",
                choices_fn=self.get_metadata_choices,
                doc="""This is the metadata tag that identifies the well name
            for the images in the current cycle. Choose <i>None</i> if
            your assay does not have metadata for the well.""")
        self.site_metadata = cps.Choice(
                "Site metadata", [], value="Site",
                choices_fn=self.get_metadata_choices,
                doc=
                """This is the metadata tag that identifies the site name
                for the images in the current cycle. Choose <i>None</i> if
                your assay doesn't divide wells up into sites or if this
                tag is not required for other reasons.""")
        self.divider = cps.Divider()
        self.wants_to_choose_measurements = cps.Binary(
                "Choose measurements?", False,
                doc="""
            This setting lets you choose between exporting all measurements or
            just the ones that you choose. Select <i>%(YES)s</i> to pick the
            measurements to be exported. Select <i>%(NO)s</i> to automatically
            export all measurements available at this stage of the pipeline.
            """ % globals())
        self.measurements = cps.MeasurementMultiChoice(
                "Measurements to export",
                doc="""
            <i>(Used only if choosing measurements.)</i>
            <br>
            This setting lets you choose individual measurements to be exported.
            Check the measurements you want to export.
            """)
        self.objects_to_export = []
        self.add_objects_button = cps.DoSomething(
                "Add objects to export", "Add objects",
                self.add_objects)
        self.images_to_export = []
        self.add_image_button = cps.DoSomething(
                "Add an image to export", "Add image",
                self.add_image)
        self.objects_count = cps.HiddenCount(self.objects_to_export)
        self.images_count = cps.HiddenCount(self.images_to_export)

    def add_objects(self, can_delete=True):
        group = cps.SettingsGroup()
        self.objects_to_export.append(group)
        group.append(
                "objects_name",
                cps.ObjectNameSubscriber(
                        "Objects name", value="Nuclei",
                        doc="""
                This setting lets you choose the objects you want to export.
                <b>ExportToCellH5</b> will write the segmentation of the objects
                to your CellH5 file so that they can be saved and used by other
                applications that support the format.
                """))
        group.append(
                "Remover",
                cps.RemoveSettingButton(
                        "Remove the objects above", "Remove",
                        self.objects_to_export, group))

    def add_image(self, can_delete=True):
        group = cps.SettingsGroup()
        self.images_to_export.append(group)
        group.append("image_name",
                     cps.ImageNameSubscriber(
                             "Image name", value="DNA",
                             doc="""
            This setting lets you choose the images you want to export.
            <b>ExportToCellH5</b> will write the image
            to your CellH5 file so that it can be used by other
            applications that support the format.
            """
                     ))
        group.append("remover",
                     cps.RemoveSettingButton(
                             "Remove the image above", "Remove",
                             self.objects_to_export, group))

    def get_metadata_choices(self, pipeline):
        columns = pipeline.get_measurement_columns(self)
        choices = [self.IGNORE_METADATA]
        for column in columns:
            object_name, feature_name, column_type = column[:3]
            if object_name == cpmeas.IMAGE and \
                    column_type.startswith(cpmeas.COLTYPE_VARCHAR) and \
                    feature_name.startswith(cpmeas.C_METADATA + "_"):
                choices.append(feature_name.split("_", 1)[1])
        return choices

    def settings(self):
        result = [
            self.objects_count, self.images_count,
            self.directory, self.file_name, self.overwrite_ok, self.repack,
            self.plate_metadata, self.well_metadata, self.site_metadata,
            self.wants_to_choose_measurements, self.measurements]
        for objects_group in self.objects_to_export:
            result += objects_group.pipeline_settings()
        for images_group in self.images_to_export:
            result += images_group.pipeline_settings()
        return result

    def visible_settings(self):
        result = [
            self.directory, self.file_name, self.overwrite_ok, self.repack,
            self.plate_metadata, self.well_metadata, self.site_metadata,
            self.divider, self.wants_to_choose_measurements]
        if self.wants_to_choose_measurements:
            result.append(self.measurements)

        for group in self.objects_to_export:
            result += group.visible_settings()
        result.append(self.add_objects_button)
        for group in self.images_to_export:
            result += group.visible_settings()
        result.append(self.add_image_button)
        return result

    def get_path_to_master_file(self, measurements):
        return os.path.join(self.directory.get_absolute_path(measurements),
                            self.file_name.value)

    def get_site_path(self, workspace, image_number):
        '''Get the plate / well / site tuple that identifies a field of view

        workspace - workspace for the analysis containing the metadata
                    measurements to be mined.

        image_number - the image number for the field of view

        returns a tuple which can be used for the hierarchical path
        to the group for a particular field of view
        '''
        m = workspace.measurements
        path = []
        for setting in self.plate_metadata, self.well_metadata, self.site_metadata:
            if setting.value == self.IGNORE_METADATA:
                path.append("NA")
            else:
                feature = "_".join((cpmeas.C_METADATA, setting.value))
                path.append(m[cpmeas.IMAGE, feature, image_number])
        return tuple(path)

    def get_subfile_name(self, workspace):
        '''Contact the UI to find the cellh5 file to use to store results

        Internally, this tells the UI to create a link from the master file
        to the plate / well / site group that will be used to store results.
        Then, the worker writes into that file.
        '''
        master_file_name = self.get_path_to_master_file(workspace.measurements)
        path = self.get_site_path(
                workspace,
                workspace.measurements.image_set_number)
        return workspace.interaction_request(
                self, master_file_name, os.getpid(), path, headless_ok=True)

    def handle_interaction(self, master_file, pid, path):
        '''Handle an analysis worker / UI interaction

        This function is used to coordinate linking a group in the master file
        with a group in a subfile that is reserved for a particular
        analysis worker. Upon entry, the worker should be sure to have
        flushed and closed its subfile.

        master_file - the master cellh5 file which has links to groups
                      for each field of view
        pid - the process ID or other unique identifier of the worker
              talking to the master
        path - The combination of (Plate, Well, Site) that should be used
               as the folder path to the data.

        returns the name of the subfile to be used. After return, the
        subfile has been closed by the UI and a link has been established
        to the group named by the path.
        '''
        master_dict = self.get_dictionary().setdefault(master_file, {})
        if pid not in master_dict:
            md_head, md_tail = os.path.splitext(master_file)
            subfile = "%s_%s%s" % (md_head, str(pid), md_tail)
            master_dict[pid] = subfile
        else:
            subfile = master_dict[pid]

        ch5_master = cellh5.cellh5write.CH5MasterFile(master_file, "a")
        try:
            ch5_master.add_link_to_coord(self._to_ch5_coord(*path), subfile)
        finally:
            ch5_master.close()

        return subfile

    def _to_ch5_coord(self, plate, well, site):
        return cellh5.CH5PositionCoordinate(plate, well, site)

    def run(self, workspace):
        m = workspace.measurements
        object_set = workspace.object_set
        #
        # get plate / well / site as tuple
        #
        path = self.get_site_path(workspace, m.image_set_number)
        subfile_name = self.get_subfile_name(workspace)

        ### create CellH5 file
        with cellh5.cellh5write.CH5FileWriter(subfile_name, mode="a") as c5_file:
            ### add Postion (==plate, well, site) triple
            c5_pos = c5_file.add_position(self._to_ch5_coord(*path))

            for ch_idx, object_group in enumerate(self.objects_to_export):
                objects_name = object_group.objects_name.value
                objects = object_set.get_objects(objects_name)
                labels = objects.segmented
                if ch_idx == 0:
                    ### get shape of 5D cube
                    shape5D = (len(self.objects_to_export), 1, 1,
                               labels.shape[0], labels.shape[1])
                    dtype5D = np.uint16

                    ### create lablel writer for incremental writing
                    c5_label_writer = c5_pos.add_label_image(shape=shape5D, dtype=dtype5D)
                    c5_label_def = cellh5.cellh5write.CH5ImageRegionDefinition()

                c5_label_writer.write(labels, c=ch_idx, t=0, z=0)
                c5_label_def.add_row(region_name=objects_name, channel_idx=ch_idx)

            if len(self.objects_to_export) > 0:
                ### finalize the writer
                c5_label_writer.write_definition(c5_label_def)
                c5_label_writer.finalize()

            n_channels = 0
            max_scale = 1
            max_i = 1
            max_j = 1
            for image_group in self.images_to_export:
                image = m.get_image(image_group.image_name.value)
                pixel_data = image.pixel_data
                if pixel_data.ndim == 3:
                    n_channels += min(pixel_data.shape[2], 3)
                else:
                    n_channels += 1
                max_scale = max(image.scale, max_scale)
                max_i = max(pixel_data.shape[0], max_i)
                max_j = max(pixel_data.shape[1], max_j)

            ### get shape of 5D cube
            shape5D = (n_channels, 1, 1, max_i, max_j)
            for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
                if max_scale <= np.iinfo(dtype).max:
                    dtype5D = dtype
                    break

            ### create image writer for incremental writing
            c5_image_writer = c5_pos.add_image(shape=shape5D, dtype=dtype5D)
            c5_image_def = cellh5.cellh5write.CH5ImageChannelDefinition()

            ch_idx = 0
            for image_group in self.images_to_export:
                image_name = image_group.image_name.value
                image = m.get_image(image_name).pixel_data
                scale = m.get_image(image_name).scale
                if not np.issubdtype(image.dtype, np.dtype(bool).type):
                    if scale == 1:
                        scale = max_scale
                    image = image * scale
                if image.ndim == 3:
                    for c in range(min(image.shape[2], 3)):
                        color_name, html_color = COLORS[c]
                        c5_image_writer.write(
                                image[:, :, c].astype(dtype5D),
                                c=ch_idx, t=0, z=0)
                        c5_image_def.add_row(
                                channel_name="_".join((image_name, color_name)),
                                description="%s %s intensity" %
                                            (image_name, color_name),
                                is_physical=True,
                                voxel_size=(1, 1, 1),
                                color=html_color)
                        ch_idx += 1
                else:
                    c5_image_writer.write(
                            image.astype(dtype5D),
                            c=ch_idx, t=0, z=0)
                    c5_image_def.add_row(
                            channel_name=image_name,
                            description=image_name,
                            is_physical=True,
                            voxel_size=(1, 1, 1),
                            color="0xFFFFFF")
                    ch_idx += 1
            c5_image_writer.write_definition(c5_image_def)
            c5_image_writer.finalize()

            columns = workspace.pipeline.get_measurement_columns(self)
            if self.wants_to_choose_measurements:
                to_keep = set([
                                  (self.measurements.get_measurement_object(s),
                                   self.measurements.get_measurement_feature(s))
                                  for s in self.measurements.selections])

                def keep(column):
                    return (column[0], column[1]) in to_keep

                columns = filter(keep, columns)
            #
            # I'm breaking the data up into the most granular form so that
            # it's clearer how it's organized. I'm expecting that you would
            # organize it differently when actually storing.
            #

            ### 0) extract object information (i.e. object_label_no)
            ### 1) extract all single cell features and write it as feature matrix (for e.g. classification)
            ### 2) extract Center
            ### 3) create artifical Bounding box... usefull for displaying it in fiji lateron
            ### 4) Don't see the point of features extracted on "Image" the only real and useful feature there is "Count" which can be deduced from single cell information

            ### 0) and 1) filter columns for cellular features
            feature_cols = filter(
                    lambda xxx: (xxx[0] not in (cpmeas.EXPERIMENT, cpmeas.IMAGE)) and
                                m.has_feature(xxx[0], xxx[1]), columns)

            ### iterate over objects to export
            for ch_idx, object_group in enumerate(self.objects_to_export):
                objects_name = object_group.objects_name.value
                objects = object_set.get_objects(objects_name)

                ### find features for that object
                feature_cols_per_object = filter(lambda xxx: xxx[0] == objects_name, feature_cols)

                c5_object_writer = c5_pos.add_region_object(objects_name)
                object_labels = objects.indices

                c5_object_writer.write(t=0, object_labels=np.array(object_labels))
                c5_object_writer.write_definition()
                c5_object_writer.finalize()

                ### iterate over all cellular feature to get feature matrix

                n_features = len(feature_cols_per_object)
                if n_features > 0:
                    feature_names = []
                    feature_matrix = []
                    for column in feature_cols_per_object:
                        object_name, feature_name = column[:2]
                        values = m[object_name, feature_name]

                        feature_names.append(feature_name)
                        feature_matrix.append(values[:, np.newaxis])

                    feature_matrix = np.concatenate(feature_matrix, axis=1)

                    c5_feature_writer = c5_pos.add_object_feature_matrix(
                            object_name=object_name,
                            feature_name="object_features",
                            n_features=n_features, dtype=np.float32)
                    c5_feature_writer.write(feature_matrix)
                    c5_feature_writer.write_definition(feature_names)
                    c5_feature_writer.finalize()

                ### iterate over Location  to create bounding_box and center
                c5_bbox = c5_pos.add_object_bounding_box(
                        object_name=objects_name)

                if objects.count > 0:
                    ijv = objects.ijv
                    min_x = scipy.ndimage.minimum(
                            ijv[:, 1], ijv[:, 2], objects.indices)
                    max_x = scipy.ndimage.maximum(
                            ijv[:, 1], ijv[:, 2], objects.indices)
                    min_y = scipy.ndimage.minimum(
                            ijv[:, 0], ijv[:, 2], objects.indices)
                    max_y = scipy.ndimage.maximum(
                            ijv[:, 0], ijv[:, 2], objects.indices)
                    location_x = scipy.ndimage.mean(
                            ijv[:, 1], ijv[:, 2], objects.indices)
                    location_y = scipy.ndimage.mean(
                            ijv[:, 0], ijv[:, 2], objects.indices)
                    bb = np.c_[min_x, max_x, min_y, max_y]
                else:
                    bb = np.zeros((0, 4))
                    location_x = np.zeros(0)
                    location_y = np.zeros(0)

                c5_bbox.write(bb.astype(np.int32))
                c5_bbox.write_definition()
                c5_bbox.finalize()

                c5_center = c5_pos.add_object_center(object_name=objects_name)
                locations = {'x': location_x, 'y': location_y}
                cent = np.column_stack(
                        [locations[axis] for axis in c5_center.dtype.names])

                c5_center.write(cent.astype(np.int32))
                c5_center.write_definition()
                c5_center.finalize()
            #
            # The last part deals with relationships between segmentations.
            # The most typical relationship is "Parent" which is explained below,
            # but you can also have things like first nearest and second nearest
            # neighbor or in tracking, the relationship between the segmentation
            # of the previous and next frames.
            #
            for key in m.get_relationship_groups():
                relationships = m.get_relationships(
                        key.module_number, key.relationship,
                        key.object_name1, key.object_name2,
                        [m.image_set_number])
                for image_number1, image_number2, \
                    object_number1, object_number2 in relationships:
                    if image_number1 == image_number2 and \
                                    key.relationship == R_PARENT:
                        #
                        # Object 1 is the parent to object 2 - this is the
                        # most common relationship, so if you can only record
                        # one, this is it. "Parent" usually means that
                        # the child's segmentation was seeded by the parent
                        # segmentation (e.g. Parent = nucleus, child = cell),
                        # but can also be something like Parent = cell,
                        # child = all organelles within the cell
                        #
                        # object_name1 is the name of the parent segmentation
                        # object_name2 is the name of the child segmentation
                        # object_number1 is the index used to label the
                        #                parent in the parent segmentation
                        # object_number2 is the index used to label the
                        #                child in the child segmentation
                        continue
                    if image_number1 != m.image_set_number:
                        path1 = self.get_site_path(workspace, image_number1)
                    else:
                        path1 = path
                    if image_number2 != m.image_set_number:
                        path2 = self.get_site_path(workspace, image_number2)
                    else:
                        path2 = path
                    #
                    # TODO: this is sort of extra credit, but the relationships
                    #       relate an object in one segmentation to another.
                    #       For tracking, these can be in different image
                    #       sets, (e.g. the cell at time T and at time T+1).
                    #       So, given object 1 and object 2, path1 and path2
                    #       tell you how the objects are related between planes.
                    pass

    def post_run(self, workspace):
        if self.repack:
            ### to be implemented with
            ### ch5_master.repack()
            return
            measurements = workspace.measurements
            fd, temp_name = tempfile.mkstemp(
                    suffix=".ch5",
                    dir=self.directory.get_absolute_path())

            master_name = self.get_path_to_master_file(workspace.measurements)
            src = h5py.File(master_name, "r")
            dest = h5py.File(temp_name)
            os.close(fd)
            for key in src:
                dest.copy(src[key], dest, expand_external=True)
            src.close()
            dest.close()
            os.unlink(master_name)
            os.rename(temp_name, master_name)

    def prepare_settings(self, setting_values):
        objects_count, images_count = [int(x) for x in setting_values[:2]]
        del self.objects_to_export[:]
        while len(self.objects_to_export) < objects_count:
            self.add_objects()
        del self.images_to_export[:]
        while len(self.images_to_export) < images_count:
            self.add_image()
