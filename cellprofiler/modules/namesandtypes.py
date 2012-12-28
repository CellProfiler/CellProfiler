'''<b>NamesAndTypes</b> - assign images to channels
<hr>
TO-DO: document module
'''

#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Copyright (c) 2003-2009 Massachusetts Institute of Technology
#Copyright (c) 2009-2012 Broad Institute
#All rights reserved.
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org

import logging
logger = logging.getLogger(__name__)

import hashlib
import numpy as np
import os
import re

import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
from cellprofiler.modules.images import FilePredicate
from cellprofiler.modules.images import ExtensionPredicate
from cellprofiler.modules.images import ImagePredicate
from cellprofiler.modules.images import DirectoryPredicate
from cellprofiler.modules.images import Images
from cellprofiler.modules.loadimages import LoadImagesImageProviderURL
from cellprofiler.modules.loadimages import convert_image_to_objects
from bioformats.formatreader import get_omexml_metadata, load_using_bioformats
import bioformats.omexml as OME

ASSIGN_ALL = "Assign all images"
ASSIGN_GUESS = "Try to guess image assignment"
ASSIGN_RULES = "Assign images matching rules"

LOAD_AS_GRAYSCALE_IMAGE = "Grayscale image"
LOAD_AS_COLOR_IMAGE = "Color image"
LOAD_AS_MASK = "Mask"
LOAD_AS_ILLUMINATION_FUNCTION = "Illumination function"
LOAD_AS_OBJECTS = "Objects"
LOAD_AS_ALL = [ LOAD_AS_GRAYSCALE_IMAGE,
                LOAD_AS_COLOR_IMAGE,
                LOAD_AS_MASK,
                LOAD_AS_ILLUMINATION_FUNCTION,
                LOAD_AS_OBJECTS]

IDX_ASSIGNMENTS_COUNT = 5

MATCH_BY_ORDER = "Order"
MATCH_BY_METADATA = "Metadata"

IMAGE_NAMES = ["DNA", "GFP", "Actin"]
OBJECT_NAMES = ["Cell", "Nucleus", "Cytoplasm", "Speckle"]

class NamesAndTypes(cpm.CPModule):
    variable_revision_number = 1
    module_name = "NamesAndTypes"
    category = "File Processing"
    
    def create_settings(self):
        self.pipeline = None
        self.ipds = []
        self.image_sets = []
        self.metadata_keys = []
        
        self.assignment_method = cps.Choice(
            "Assignment method", [ASSIGN_ALL, ASSIGN_GUESS, ASSIGN_RULES],
            doc = """How do you want to assign images to channels?<br>
            This setting controls how different types (e.g. an image
            of the GFP stain and a brightfield image) are assigned different
            names so that each type can be treated differently by
            downstream modules. There are three choices:<br>
            <ul><li><b>%(ASSIGN_ALL)s</b> - give every image the same name.
            This is the simplest choice and the appropriate one if you have
            only one kind of image (or only one image). CellProfiler will
            give each image the same name and the pipeline will load only
            one of the images per iteration.</li>
            <li><b>%(ASSIGN_GUESS)s</b> - CellProfiler will guess the image
            assignment and will display the results of the guess which
            you can accept if correct.</li>
            <li><b>%(ASSIGN_RULES)s</b> - give images one of several names
            depending on the file name, directory and metadata. This is the
            appropriate choice if more than one image was taken of each 
            imaging site. You will be asked for distinctive criteria for
            each image and will be able to assign each category of image
            a name that can be referred to in downstream modules.</li></ul>
            """ % globals())
        
        self.single_load_as_choice = cps.Choice(
            "Load as", [ LOAD_AS_GRAYSCALE_IMAGE,
                         LOAD_AS_COLOR_IMAGE,
                         LOAD_AS_MASK])
        
        self.single_image_provider = cps.FileImageNameProvider(
            "Image name", IMAGE_NAMES[0])
            
        self.assignments = []
        self.assignments_count = cps.HiddenCount( self.assignments,
                                                  "Assignments count")
        self.add_assignment(can_remove = False)
        self.add_assignment_button = cps.DoSomething(
            "Add another assignment rule", "Add",
            self.add_assignment)
        self.matching_choice = cps.Choice(
            "Assign channels by",
            [MATCH_BY_ORDER, MATCH_BY_METADATA],
            doc = """How do you want to match the image from one channel with
            the images from other channels?
            <p>This setting controls how CellProfiler picks which images
            should be matched together when analyzing all of the images
            from one site. You can either match by order or match using
            metadata.<p>
            <i>Match by order</i>: CellProfiler will order the images in
            each channel alphabetically by their file path name and, for movies
            or TIF stacks, will order the frames by their order in the file.
            CellProfiler will then match the first from one channel to the
            first from another channel. This is simple when it works, but
            will match the wrong files if files are missing or misnamed.<p>
            <i>Match by metadata</i>: CellProfiler will match files with
            the same metadata values. For instance, if both files have
            well and site metadata, CellProfiler will match the file
            in one channel with well A01 and site 1 with the file in the
            other channel with well A01 and site 1 and so on. CellProfiler
            can match a single file for one channel against many files from
            another channel, for instance an illumination correction file
            for an entire plate against every image file for that plate.
            <i>Match by metadata</i> is powerful, less prone to inadvertent
            errors, but harder to use than <i>Match by order</i>.
            """)
        self.join = cps.Joiner("")
        self.update_table_button = cps.DoSomething(
            "","Update the table below", self.make_image_sets_and_update_table)
        self.table = cps.Table("")
        
    def add_assignment(self, can_remove = True):
        '''Add a rules assignment'''
        group = cps.SettingsGroup()
        self.assignments.append(group)
        
        if can_remove:
            group.append("divider", cps.Divider())
        
        mp = MetadataPredicate("Metadata", "Have %s matching", 
                               doc="Has metadata matching the value you enter")
        mp.set_metadata_keys(self.metadata_keys)
        group.append("rule_filter", cps.Filter(
            "Match this rule",
            [FilePredicate(),
             DirectoryPredicate(),
             ExtensionPredicate(),
             ImagePredicate(),
             mp],
            'or (file does contain "")'))
        
        unique_image_name = None
        all_image_names = [
            other_group.image_name for other_group in self.assignments[:-1]]
        for image_name in IMAGE_NAMES:
            if image_name not in all_image_names:
                unique_image_name = image_name
                break
        else:
            for i in xrange(1, 1000):
                image_name = "Channel%d" % i
                if image_name not in all_image_names:
                    unique_image_name = image_name
                    break
                    
        group.append("image_name", cps.FileImageNameProvider(
            "Image name", unique_image_name))
        
        unique_object_name = None
        all_object_names = [
            other_group.object_name for other_group in self.assignments[:-1]]
        for object_name in OBJECT_NAMES:
            if object_name not in all_object_names:
                unique_object_name = object_name
                break
        else:
            for i in xrange(1, 1000):
                object_name = "Object%d" % i
                if object_name not in all_object_names:
                    unique_object_name = object_name
                    break

        group.append("object_name", cps.ObjectNameProvider(
            "Objects name", unique_object_name))
        
        group.append("load_as_choice", cps.Choice(
            "Load as", LOAD_AS_ALL))
        
        group.can_remove = can_remove
        if can_remove:
            def do_remove():
                self.assignments.remove(group)
                self.update_all_columns()
                
            group.append("remover", cps.DoSomething(
                'Remove above rule', "Remove", do_remove))
        if self.pipeline is not None:
            self.update_all_columns()
            
    def settings(self):
        result = [self.assignment_method, self.single_load_as_choice,
                  self.single_image_provider, self.join, self.matching_choice,
                  self.assignments_count]
        for assignment in self.assignments:
            result += [assignment.rule_filter, assignment.image_name,
                       assignment.object_name, assignment.load_as_choice]
        return result
    
    def visible_settings(self):
        result = [self.assignment_method]
        if self.assignment_method == ASSIGN_ALL:
            result += [self.single_load_as_choice, self.single_image_provider]
        elif self.assignment_method == ASSIGN_RULES:
            for assignment in self.assignments:
                if assignment.can_remove:
                    result += [assignment.divider]
                result += [assignment.rule_filter]
                if assignment.load_as_choice == LOAD_AS_OBJECTS:
                    result += [assignment.object_name]
                else:
                    result += [assignment.image_name]
                result += [assignment.load_as_choice]
                if assignment.can_remove:
                    result += [assignment.remover]
            result += [self.add_assignment_button]
            if len(self.assignments) > 1:
                result += [self.matching_choice]
                if self.matching_choice == MATCH_BY_METADATA:
                    result += [self.join]
        result += [self.update_table_button]
        if len(self.table.data) > 0:
            self.update_table_button.label = "Update table"
            result += [self.table]
        else:
            self.update_table_button.label = "Show table"
        return result
    
    def prepare_settings(self, setting_values):
        n_assignments = int(setting_values[IDX_ASSIGNMENTS_COUNT])
        if len(self.assignments) > n_assignments:
            del self.assignments[n_assignments:]
        while len(self.assignments) < n_assignments:
            self.add_assignment()
            
    def post_pipeline_load(self, pipeline):
        '''Fix up metadata predicates after the pipeline loads'''
        if self.assignment_method == ASSIGN_RULES:
            filters = []
            self.metadata_keys = []
            for group in self.assignments:
                rules_filter = group.rule_filter
                filters.append(rules_filter)
                assert isinstance(rules_filter, cps.Filter)
                #
                # The problem here is that the metadata predicates don't
                # know what possible metadata keys are allowable and
                # that computation could be (insanely) expensive. The
                # hack is to scan for the string we expect in the
                # raw text.
                #
                # The following looks for "(metadata does <kwd>" or
                # "(metadata doesnot <kwd>"
                #
                # This isn't perfect, of course, but it is enough to get
                # the filter's text to parse if the text is valid.
                #
                pattern = r"\(%s (?:%s|%s) ((?:\\.|[^ )])+)" % \
                (MetadataPredicate.SYMBOL, 
                 cps.Filter.DoesNotPredicate.SYMBOL,
                 cps.Filter.DoesPredicate.SYMBOL)
                text = rules_filter.value_text
                while True:
                    match = re.search(pattern, text)
                    if match is None:
                        break
                    key = cps.Filter.FilterPredicate.decode_symbol(
                        match.groups()[0])
                    self.metadata_keys.append(key)
                    text = text[match.end():]
            self.metadata_keys = list(set(self.metadata_keys))
            for rules_filter in filters:
                for predicate in rules_filter.predicates:
                    if isinstance(predicate, MetadataPredicate):
                        predicate.set_metadata_keys(self.metadata_keys)
                        
    def is_load_module(self):
        return True
    
    def change_causes_prepare_run(self, setting):
        '''Return True if changing the setting passed changes the image sets
        
        setting - the setting that was changed
        '''
        return setting in self.settings()
    
    def prepare_run(self, workspace):
        '''Write the image set to the measurements'''
        if workspace.pipeline.in_batch_mode():
            return True
        try:
            old_pipeline = self.pipeline
            self.pipeline = workspace.pipeline
            self.ipds = self.pipeline.get_filtered_image_plane_details(True)
            self.metadata_keys = set()
            for ipd in self.ipds:
                self.metadata_keys.update(ipd.metadata.keys())
            self.update_all_columns()
            self.make_image_sets()
            self.pipeline = old_pipeline
            
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            image_sets = []
            column_names = self.column_names
            for keys, ipds in self.image_sets:
                if any([len(ipds.get(column_name, tuple())) != 1
                        for column_name in column_names]):
                    logger.info("Skipping image set %s - no or multiple matches for some image" % repr(keys))
                    continue
                image_sets.append((keys, ipds))
            if len(image_sets) == 0:
                return False
            
            image_numbers = range(1, len(image_sets) + 1)
            m.add_all_measurements(cpmeas.IMAGE, cpmeas.IMAGE_NUMBER, 
                                   image_numbers)
            
            metadata_key_names = self.get_metadata_column_names()
            if self.assignment_method == ASSIGN_ALL:
                load_choices = [self.single_load_as_choice.value]
            elif self.assignment_method == ASSIGN_RULES:
                load_choices = [ group.load_as_choice.value
                                 for group in self.assignments]
                if self.matching_choice == MATCH_BY_METADATA:
                    metadata_feature_names = [
                        '_'.join((cpmeas.C_METADATA, n))
                        for n in metadata_key_names]
                    m.set_metadata_tags(metadata_feature_names)
                    key_columns = [[] for f in metadata_feature_names]
                    for keys, ipds in image_sets:
                        for key_column, key in zip(key_columns, keys):
                            key_column.append(key)
                    for metadata_feature_name, key_column in zip(
                        metadata_feature_names, key_columns):
                        m.add_all_measurements(
                            cpmeas.IMAGE, metadata_feature_name,
                            key_column)
                    
            ImageSetChannelDescriptor = workspace.pipeline.ImageSetChannelDescriptor
            d = { 
                LOAD_AS_COLOR_IMAGE: ImageSetChannelDescriptor.CT_COLOR,
                LOAD_AS_GRAYSCALE_IMAGE: ImageSetChannelDescriptor.CT_GRAYSCALE,
                LOAD_AS_ILLUMINATION_FUNCTION: ImageSetChannelDescriptor.CT_FUNCTION,
                LOAD_AS_MASK: ImageSetChannelDescriptor.CT_MASK,
                LOAD_AS_OBJECTS: ImageSetChannelDescriptor.CT_OBJECTS }
            iscds = [ImageSetChannelDescriptor(column_name, d[load_choice])
                     for column_name, load_choice in zip(column_names, load_choices)]
            m.set_channel_descriptors(iscds)
            
            ipd_columns = [[] for column_name in column_names]
            
            for keys, ipds in image_sets:
                for column, column_name in zip(ipd_columns, column_names):
                    column.append(ipds[column_name][0])
            for iscd, ipds in zip(iscds, ipd_columns):
                if iscd.channel_type == ImageSetChannelDescriptor.CT_OBJECTS:
                    url_category = cpmeas.C_OBJECTS_URL
                    path_name_category = cpmeas.C_OBJECTS_PATH_NAME
                    file_name_category = cpmeas.C_OBJECTS_FILE_NAME
                    series_category = cpmeas.C_OBJECTS_SERIES
                    frame_category = cpmeas.C_OBJECTS_FRAME
                    channel_category = cpmeas.C_OBJECTS_CHANNEL
                else:
                    url_category = cpmeas.C_URL
                    path_name_category = cpmeas.C_PATH_NAME
                    file_name_category = cpmeas.C_FILE_NAME
                    series_category = cpmeas.C_SERIES
                    frame_category = cpmeas.C_FRAME
                    channel_category = cpmeas.C_CHANNEL
                url_feature, path_name_feature, file_name_feature,\
                    series_feature, frame_feature, channel_feature = [
                        "%s_%s" % (category, iscd.name) for category in (
                            url_category, path_name_category, file_name_category,
                            series_category, frame_category, channel_category)]
                m.add_all_measurements(cpmeas.IMAGE, url_feature,
                                  [ipd.url for ipd in ipds])
                m.add_all_measurements(
                    cpmeas.IMAGE, path_name_feature,
                    [os.path.split(ipd.path)[0] for ipd in ipds])
                m.add_all_measurements(
                    cpmeas.IMAGE, file_name_feature,
                    [os.path.split(ipd.path)[1] for ipd in ipds])
                all_series = [ipd.series for ipd in ipds]
                if any([x is not None for x in all_series]):
                    m.add_all_measurements(
                        cpmeas.IMAGE, series_feature, all_series)
                all_frames = [ipd.index for ipd in ipds]
                if any([x is not None for x in all_frames]):
                    m.add_all_measurements(
                        cpmeas.IMAGE, frame_feature, all_frames)
                all_channels = [ipd.channel for ipd in ipds]
                if any([x is not None for x in all_channels]):
                    m.add_all_measurements(
                        cpmeas.IMAGE, channel_feature, all_channels)
            #
            # Scan through each image set's metadata to get a set of metadata
            # columns. Every image set will have values for each
            # metadata column.
            #
            md_dict = {}
            #
            # Populate the dictionary using the first image set
            #
            for ipd_column in ipd_columns:
                for key, value in ipd_column[0].metadata.iteritems():
                    if not md_dict.has_key(key):
                        md_dict[key] = set()
                    md_dict[key].add(value)
            #
            # Find all metadata items that are singly valued.
            # Recreate the dictionary as key: list of one value element
            #
            md_dict = dict([ (k, [v.pop()]) for k, v in md_dict.iteritems()
                             if len(v) == 1])
            for i in range(1, len(ipd_columns[0])):
                image_set_metadata = {}
                for ipd_column in ipd_columns:
                    metadata = ipd_column[i].metadata
                    for key in md_dict:
                        if key in metadata:
                            if key in image_set_metadata:
                                if image_set_metadata[key] != metadata[key]:
                                    image_set_metadata[key] = False
                            else:
                                image_set_metadata[key] = metadata[key]
                for key in md_dict.keys():
                    if ((key not in image_set_metadata) or 
                        image_set_metadata is False):
                        del md_dict[key]
                    else:
                        md_dict[key].append(image_set_metadata[key])
            #
            # Populate the metadata measurements
            #
            for name, values in md_dict.iteritems():
                feature_name = "_".join((cpmeas.C_METADATA, name))
                m.add_all_measurements(cpmeas.IMAGE,
                                       feature_name,
                                       values)
            return True
        finally:
            self.on_deactivated()
            
    def prepare_to_create_batch(self, workspace, fn_alter_path):
        '''Alter pathnames in preparation for batch processing
        
        workspace - workspace containing pipeline & image measurements
        fn_alter_path - call this function to alter any path to target
                        operating environment
        '''
        if self.assignment_method == ASSIGN_ALL:
            names = [self.single_image_provider.value]
            is_image = [True]
        else:
            names = []
            is_image = []
            for group in self.assignments:
                if group.load_as_choice == LOAD_AS_OBJECTS:
                    names.append(group.object_name.value)
                    is_image.append(False)
                else:
                    names.append(group.image_name.value)
                    is_image.append(True)
        for name, iz_image in zip(names, is_image):
            workspace.measurements.alter_path_for_create_batch(
                name, iz_image, fn_alter_path)
                        
    def is_input_module(self):
        return True
            
    def run(self, workspace):
        if self.assignment_method == ASSIGN_ALL:
            name = self.single_image_provider.value
            load_choice = self.single_load_as_choice.value
            self.add_image_provider(workspace, name, load_choice)
        else:
            for group in self.assignments:
                if group.load_as_choice == LOAD_AS_OBJECTS:
                    self.add_objects(workspace, group.object_name.value)
                else:
                    self.add_image_provider(workspace, 
                                            group.image_name.value,
                                            group.load_as_choice.value)
            
    
    def add_image_provider(self, workspace, name, load_choice):
        '''Put an image provider into the image set
        
        workspace - current workspace
        name - name of the image
        load_choice - one of the LOAD_AS_... choices
        '''
        def fetch_measurement_or_none(category):
            feature = category + "_" + name
            if workspace.measurements.has_feature(cpmeas.IMAGE, feature):
                return workspace.measurements.get_measurement(
                    cpmeas.IMAGE, feature)
            else:
                return None
            
        url = fetch_measurement_or_none(cpmeas.C_URL).encode("utf-8")
        series = fetch_measurement_or_none(cpmeas.C_SERIES)
        index = fetch_measurement_or_none(cpmeas.C_FRAME)
        channel = fetch_measurement_or_none(cpmeas.C_CHANNEL)
        
        if load_choice == LOAD_AS_COLOR_IMAGE:
            provider = ColorImageProvider(name, url, series, index)
        elif load_choice == LOAD_AS_GRAYSCALE_IMAGE:
            provider = MonochromeImageProvider(name, url, series, index,
                                               channel)
        elif load_choice == LOAD_AS_ILLUMINATION_FUNCTION:
            provider = MonochromeImageProvider(name, url, series, index, 
                                               channel, False)
        elif load_choice == LOAD_AS_MASK:
            provider = MaskImageProvider(name, url, series, index, channel)
        workspace.image_set.providers.append(provider)
        self.add_provider_measurements(provider, workspace.measurements, \
                                       cpmeas.IMAGE)
        
    @staticmethod
    def add_provider_measurements(provider, m, image_or_objects):
        '''Add image measurements using the provider image and file
        
        provider - an image provider: get the height and width of the image
                   from the image pixel data and the MD5 hash from the file
                   itself.
        m - measurements structure
        image_or_objects - cpmeas.IMAGE if the provider is an image provider
                           otherwise cpmeas.OBJECT if it provides objects
        '''
        from cellprofiler.modules.loadimages import \
             C_MD5_DIGEST, C_SCALING, C_HEIGHT, C_WIDTH
        
        name = provider.get_name()
        m[cpmeas.IMAGE, C_MD5_DIGEST + "_" + name] = \
            NamesAndTypes.get_file_hash(provider)
        img = provider.provide_image(m)
        m[cpmeas.IMAGE, C_WIDTH + "_" + name] = img.pixel_data.shape[1]
        m[cpmeas.IMAGE, C_HEIGHT + "_" + name] = img.pixel_data.shape[0]
        if image_or_objects == cpmeas.IMAGE:
            m[cpmeas.IMAGE, C_SCALING + "_" + name] = provider.scale
        
    @staticmethod
    def get_file_hash(provider):
        '''Get an md5 checksum from the (cached) file courtesy of the provider'''
        hasher = hashlib.md5()
        with open(provider.get_full_name(), "rb") as fd:
            while True:
                buf = fd.read(65536)
                if len(buf) == 0:
                    break
                hasher.update(buf)
        return hasher.hexdigest()
    
    def add_objects(self, workspace, name):
        '''Add objects loaded from a file to the object set'''
        from cellprofiler.modules.identify import add_object_count_measurements
        from cellprofiler.modules.identify import add_object_location_measurements
        from cellprofiler.modules.identify import add_object_location_measurements_ijv
        
        def fetch_measurement_or_none(category):
            feature = category + "_" + name
            if workspace.measurements.has_feature(cpmeas.IMAGE, feature):
                return workspace.measurements.get_measurement(
                    cpmeas.IMAGE, feature)
            else:
                return None
            
        url = fetch_measurement_or_none(cpmeas.C_OBJECTS_URL).encode("utf-8")
        series = fetch_measurement_or_none(cpmeas.C_OBJECTS_SERIES)
        index = fetch_measurement_or_none(cpmeas.C_OBJECTS_FRAME)
        provider = ObjectsImageProvider(name, url, series, index)
        self.add_provider_measurements(provider, workspace.measurements, 
                                       cpmeas.OBJECT)
        image = provider.provide_image(workspace.image_set)
        o = cpo.Objects()
        if image.pixel_data.shape[2] == 1:
            o.segmented = image.pixel_data[:, :, 0]
            add_object_location_measurements(workspace.measurements,
                                             name,
                                             o.segmented,
                                             o.count)
        else:
            ijv = np.zeros((0, 3), int)
            for i in range(image.pixel_data.shape[2]):
                plane = image.pixel_data[:, :, i]
                shape = plane.shape
                i, j = np.mgrid[0:shape[0], 0:shape[1]]
                ijv = np.vstack(
                    (ijv, 
                     np.column_stack([x[plane != 0] for x in (i, j, plane)])))
            o.set_ijv(ijv, shape)
            add_object_location_measurements_ijv(workspace.measurements,
                                                 name, o.ijv, o.count)
        add_object_count_measurements(workspace.measurements, name, o.count)
        workspace.object_set.add_objects(o, name)
                     
    def on_activated(self, workspace):
        self.pipeline = workspace.pipeline
        self.pipeline.load_image_plane_details(workspace)
        self.ipds = self.pipeline.get_filtered_image_plane_details(with_metadata=True)
        self.metadata_keys = set()
        for ipd in self.ipds:
            self.metadata_keys.update(ipd.metadata.keys())
        self.update_all_metadata_predicates()
        if self.join in self.visible_settings():
            self.update_all_columns()
        else:
            self.ipd_columns = []
        self.table.clear_rows()
        
    def on_deactivated(self):
        self.pipeline = None
        
    def on_setting_changed(self, setting, pipeline):
        '''Handle updates to all settings'''
        if setting.key() == self.assignment_method.key():
            self.update_all_columns()
        elif self.assignment_method == ASSIGN_RULES:
            self.update_all_metadata_predicates()
            if len(self.ipd_columns) != len(self.assignments):
                self.update_all_columns()
            else:
                for i, group in enumerate(self.assignments):
                    if setting in (group.rule_filter, group.image_name,
                                   group.object_name):
                        if setting == group.rule_filter:
                            self.ipd_columns[i] = self.filter_column(group)
                            self.update_column_metadata(i)
                            self.update_joiner()
                        else:
                            if setting == group.image_name:
                                name = group.image_name.value
                            elif setting == group.object_name:
                                name = group.object_name.value
                            else:
                                return
                            #
                            # The column was renamed.
                            #
                            old_name = self.column_names[i]
                            if old_name == name:
                                return
                            self.column_names[i] = name
                            if old_name in self.column_names:
                                # duplicate names - update the whole thing
                                self.update_joiner()
                                return
                            self.join.entities[name] = \
                                self.column_metadata_choices[i]
                            del self.join.entities[old_name]
                            joins = self.join.parse()
                            if len(joins) > 0:
                                for join in joins:
                                    join[name] = join[old_name]
                                    del join[old_name]
                            self.join.build(str(joins))
                        return
        
    def update_all_metadata_predicates(self):
        if self.assignment_method == ASSIGN_RULES:
            for group in self.assignments:
                rules_filter = group.rule_filter
                for predicate in rules_filter.predicates:
                    if isinstance(predicate, MetadataPredicate):
                        predicate.set_metadata_keys(self.metadata_keys)
                        
    def update_all_columns(self):
        if self.assignment_method == ASSIGN_ALL:
            self.ipd_columns = [ list(self.ipds)]
            column_name = self.single_image_provider.value
            self.column_names = [ column_name ]
        else:
            self.ipd_columns = [self.filter_column(group) 
                                for group in self.assignments]
            self.column_metadata_choices = [[]] * len(self.ipd_columns)
            self.column_names = [
                group.object_name.value if group.load_as_choice == LOAD_AS_OBJECTS
                else group.image_name.value for group in self.assignments]
            for i in range(len(self.ipd_columns)):
                self.update_column_metadata(i)
        self.update_all_metadata_predicates()
        self.update_joiner()
        
    def make_image_sets_and_update_table(self):
        self.update_all_columns()
        self.make_image_sets()
        self.update_table()
        
    def make_image_sets(self):
        '''Create image sets from the ipd columns and joining rules
        
        Each image set is a dictionary whose keys are column names and
        whose values are lists of ipds that match the metadata for the
        image set (hopefully a list with a single element).
        '''
        if self.assignment_method == ASSIGN_ALL:
            #
            # The keys are the image set numbers, the image sets have
            # the single column name
            #
            column_name = self.column_names[0]
            self.image_sets = [((i+1, ), { column_name: (ipd,) })
                               for i, ipd in enumerate(self.ipd_columns[0])]
        elif self.assignment_method == ASSIGN_RULES:
            #
            # d is a nested dictionary. The key is either None (= match all)
            # or the first metadata value. The value is either a subdictionary
            # or the leaf which is an image_set dictionary.
            #
            def cmp_ipds(i1, i2):
                '''Sort by path name lexical order, then by predefined order'''
                r = cmp(i1.path, i2.path)
                if r != 0:
                    return r
                return cmp(i1, i2)
            for ipd_column in self.ipd_columns:
                ipd_column.sort(cmp=cmp_ipds)
              
            if (len(self.column_names) == 1 or 
                self.matching_choice == MATCH_BY_ORDER):
                self.image_sets = []
                n_image_sets = reduce(max, [len(x) for x in self.ipd_columns], 0)
                for i in range(n_image_sets):
                    d = {}
                    for column_name, ipd_column \
                        in zip(self.column_names, self.ipd_columns):
                        if i < len(ipd_column):
                            d[column_name] = (ipd_column[i],)
                        else:
                            d[column_name] = tuple()
                            
                    self.image_sets.append(((str(i+1), ), d))
                return
            try:
                joins = self.join.parse()
                if len(joins) == 0:
                    raise ValueError("No joining criteria")
            except Exception, e:
                # Bad format for joiner
                logger.warn("Bad join format: %s" % str(e))
                self.image_sets = []
                return
            d = {}
            dd = d
            for join in joins:
                ddd = {}
                dd[None] = ddd
                dd = ddd
                
            def deep_copy_dictionary(ddd):
                '''Create a deep copy of a dictionary'''
                if all([isinstance(v, dict) for v in ddd.values()]):
                    return dict([(k, deep_copy_dictionary(v))
                                 for k, v in ddd.iteritems()])
                else:
                    return dict([(k, list(v)) for k, v in ddd.iteritems()])
                
            def assign_ipd_to_dictionary(ipd, column_name, keys, ddd):
                '''Peel off a key, find its metadata and apply to dictionaries
                
                '''
                if len(keys) == 0:
                    if not ddd.has_key(column_name):
                        ddd[column_name] = []
                    ddd[column_name].append(ipd)
                else:
                    k0 = keys[0]
                    keys = keys[1:]
                    if k0 is None:
                        #
                        # IPD is distributed to all image sets for
                        # the join
                        #
                        for k in ddd.keys():
                            assign_ipd_to_dictionary(ipd, column_name, keys,
                                                     ddd[k])
                    else:
                        m0 = ipd.metadata[k0]
                        if not ddd.has_key(m0):
                            #
                            # There is no match. Replicate the dictionary tree
                            # in the "None" category and distribute to it.
                            #
                            dcopy = deep_copy_dictionary(ddd[None])
                            ddd[m0] = dcopy
                        assign_ipd_to_dictionary(ipd, column_name, keys, ddd[m0])
                        
            for ipds, column_name in zip(self.ipd_columns, self.column_names):
                metadata_keys = [join.get(column_name) for join in joins]
                for ipd in ipds:
                    assign_ipd_to_dictionary(ipd, column_name,
                                             metadata_keys, d)
            #
            # Flatten d
            #
            def flatten_dictionary(ddd):
                '''Make a list of metadata values and image set dictionaries
                
                ddd: a dictionary of one of two types. If the values are all
                     dictionaries, then the keys are metadata values and the
                     values are either image set dictionaries or a sub-dictionary.
                     Otherwise, the dictionary is an image set dictionary.
                '''
                if all([isinstance(v, dict) for v in ddd.values()]):
                    flattened_subs = [
                        (k, flatten_dictionary(v))
                        for k, v in ddd.iteritems()
                        if k is not None]
                    combined_keys = [
                        [ (tuple([k] + list(keys)), image_set)
                          for keys, image_set in subs]
                        for k, subs in flattened_subs]
                    return sum(combined_keys, [])
                else:
                    #
                    # The dictionary is an image set, so its keyset is empty
                    #
                    return [(tuple(), ddd)]
            result = flatten_dictionary(d)
            self.image_sets = sorted(result, lambda a, b: cmp(a[0], b[0]))
        else:
            logger.warn("Unimplemented assignment method: %s" %
                        self.assignment_method.value)
            self.image_sets = []
            
    def get_image_names(self):
        '''Return the names of all images produced by this module'''
        if self.assignment_method == ASSIGN_ALL:
            return [self.single_image_provider.value]
        elif self.assignment_method == ASSIGN_RULES:
            return [group.image_name.value 
                    for group in self.assignments
                    if group.load_as_choice != LOAD_AS_OBJECTS]
        return []
    
    def get_object_names(self):
        '''Return the names of all objects produced by this module'''
        if self.assignment_method == ASSIGN_RULES:
            return [group.object_name.value
                    for group in self.assignments
                    if group.load_as_choice == LOAD_AS_OBJECTS]
        return []
        
            
    def get_measurement_columns(self, pipeline):
        '''Create a list of measurements produced by this module
        
        For NamesAndTypes, we anticipate that the pipeline will create
        the text measurements for the images.
        '''
        from cellprofiler.modules.loadimages import \
             C_FILE_NAME, C_PATH_NAME, C_URL, C_MD5_DIGEST, C_SCALING, \
             C_HEIGHT, C_WIDTH, C_SERIES, C_FRAME, \
             C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME, C_OBJECTS_URL
        from cellprofiler.modules.identify import C_NUMBER, C_COUNT, \
             C_LOCATION, FTR_OBJECT_NUMBER, FTR_CENTER_X, FTR_CENTER_Y, \
             get_object_measurement_columns
        
        image_names = self.get_image_names()
        object_names = self.get_object_names()
        result = []
        for image_name in image_names:
            result += [ (cpmeas.IMAGE, 
                         "_".join([category, image_name]),
                         coltype)
                        for category, coltype in (
                            (C_FILE_NAME, cpmeas.COLTYPE_VARCHAR_FILE_NAME),
                            (C_PATH_NAME, cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                            (C_URL, cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                            (C_MD5_DIGEST, cpmeas.COLTYPE_VARCHAR_FORMAT % 32),
                            (C_SCALING, cpmeas.COLTYPE_FLOAT),
                            (C_WIDTH, cpmeas.COLTYPE_INTEGER),
                            (C_HEIGHT, cpmeas.COLTYPE_INTEGER),
                            (C_SERIES, cpmeas.COLTYPE_INTEGER),
                            (C_FRAME, cpmeas.COLTYPE_INTEGER)
                        )]
        for object_name in object_names:
            result += [ (cpmeas.IMAGE,
                         "_".join([category, object_name]),
                         coltype)
                        for category, coltype in (
                            (C_OBJECTS_FILE_NAME, cpmeas.COLTYPE_VARCHAR_FILE_NAME),
                            (C_OBJECTS_PATH_NAME, cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                            (C_OBJECTS_URL, cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                            (C_COUNT, cpmeas.COLTYPE_INTEGER),
                            (C_MD5_DIGEST, cpmeas.COLTYPE_VARCHAR_FORMAT % 32),
                            (C_WIDTH, cpmeas.COLTYPE_INTEGER),
                            (C_HEIGHT, cpmeas.COLTYPE_INTEGER),
                            (C_SERIES, cpmeas.COLTYPE_INTEGER),
                            (C_FRAME, cpmeas.COLTYPE_INTEGER)
                            )]
            result += get_object_measurement_columns(object_name)
                            
        return result
        
    def get_categories(self, pipeline, object_name):
        from cellprofiler.modules.loadimages import \
             C_FILE_NAME, C_PATH_NAME, C_URL, C_MD5_DIGEST, C_SCALING, \
             C_HEIGHT, C_WIDTH, C_SERIES, C_FRAME, \
             C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME, C_OBJECTS_URL
        from cellprofiler.modules.identify import C_LOCATION, C_NUMBER
        result = []
        if object_name == cpmeas.IMAGE:
            has_images = any(self.get_image_names())
            has_objects = any(self.get_object_names())
            if has_images:
                result += [C_FILE_NAME, C_PATH_NAME, C_URL]
            if has_objects:
                result += [C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME, C_COUNT]
        elif object_name in self.get_object_names():
            result += [C_LOCATION, C_NUMBER]
        return result
    
    def get_measurements(self, pipeline, object_name, category):
        from cellprofiler.modules.loadimages import \
             C_FILE_NAME, C_PATH_NAME, C_URL, C_MD5_DIGEST, C_SCALING, \
             C_HEIGHT, C_WIDTH, C_SERIES, C_FRAME, \
             C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME, C_OBJECTS_URL
        from cellprofiler.modules.identify import C_NUMBER, C_COUNT, \
             C_LOCATION, FTR_OBJECT_NUMBER, FTR_CENTER_X, FTR_CENTER_Y
        
        image_names = self.get_image_names()
        object_names = self.get_object_names()
        if object_name == cpmeas.IMAGE:
            if category in (C_FILE_NAME, C_PATH_NAME, C_URL):
                return image_names
            elif category == C_COUNT:
                return object_names
        elif object_name in self.get_object_names():
            if category == C_NUMBER:
                return [FTR_OBJECT_NUMBER]
            elif category == C_LOCATION:
                return [FTR_CENTER_X, FTR_CENTER_Y]
        return []
            
    class FakeModpathResolver(object):
        '''Resolve one modpath to one ipd'''
        def __init__(self, modpath, ipd):
            self.modpath = modpath
            self.ipd = ipd
            
        def get_image_plane_details(self, modpath):
            assert len(modpath) == len(self.modpath)
            assert all([m1 == m2 for m1, m2 in zip(self.modpath, modpath)])
            return self.ipd
        
    def filter_ipd(self, ipd, group):
        modpath = Images.url_to_modpath(ipd.url)
        try:
            match = group.rule_filter.evaluate(
                (cps.FileCollectionDisplay.NODE_IMAGE_PLANE, 
                 modpath, self.FakeModpathResolver(modpath, ipd)))
            return match
        except:
            return False
        
    def filter_column(self, group):
        '''Filter all IPDs using the values specified in the group
        
        return a collection of IPDs passing the filter
        '''
        try:
            return [ipd for ipd in self.ipds
                    if self.filter_ipd(ipd, group)]
        except:
            return []
    
    def update_column_metadata(self, idx):
        '''Populate the column metadata choices with the common metadata keys
        
        Scan the IPDs for the column and find metadata keys that are common
        to all.
        '''
        if len(self.ipd_columns[idx]) == 0:
            self.column_metadata_choices[idx] = []
        else:
            keys = set(self.ipd_columns[idx][0].metadata.keys())
            for ipd in self.ipd_columns[idx][1:]:
                keys.intersection_update(ipd.metadata.keys())
            self.column_metadata_choices[idx] = list(keys)
            
    def update_joiner(self):
        '''Update the joiner setting's entities'''
        if self.assignment_method == ASSIGN_RULES:
            self.join.entities = dict([
                (column_name, column_metadata_choices)
                for column_name, column_metadata_choices 
                in zip(self.column_names, self.column_metadata_choices)])
            try:
                joins = self.join.parse()
                if len(joins) > 0:
                    for join in joins:
                        best_value = None
                        for key in join.keys():
                            if key not in self.column_names:
                                del join[key]
                            elif join[key] is not None and best_value is None:
                                best_value = join[key]
                        for i, column_name in enumerate(self.column_names):
                            if not join.has_key(column_name):
                                if best_value in self.column_metadata_choices[i]:
                                    join[column_name] = best_value
                                else:
                                    join[column_name] = None
                self.join.build(repr(joins))
            except:
                pass # bad field value
    
    def get_metadata_column_names(self):
        if (self.assignment_method == ASSIGN_RULES and 
            self.matching_choice == MATCH_BY_METADATA and
            len(self.column_names) > 1):
            joins = self.join.parse()
            metadata_columns = [
                " / ".join(set([k for k in join.values() if k is not None]))
                for join in joins]
        else:
            metadata_columns = [cpmeas.IMAGE_NUMBER]
        return metadata_columns
        
    def update_table(self):
        '''Update the table to show the current image sets'''
        joins = self.join.parse()
        self.table.clear_columns()
        self.table.clear_rows()
        
        metadata_columns = self.get_metadata_column_names()
        for i, name in enumerate(metadata_columns):
            self.table.insert_column(i, name)
        f_pathname = "Pathname: %s"
        f_filename = "Filename: %s"
        f_frame = "Frame: %s"
        f_series = "Series: %s"
        has_frame_numbers = {}
        has_series = {}
        column_counts = {}
        idx = len(metadata_columns)
        for column_name in self.column_names:
            self.table.insert_column(idx, f_pathname % column_name)
            self.table.insert_column(idx+1, f_filename % column_name)
            idx += 2
            hfn = None
            hs = None
            column_counts[column_name] = 2
            has_frame_numbers[column_name] = False
            has_series[column_name] = False

            for (_, image_set) in self.image_sets:
                for ipd in image_set.get(column_name, []):
                    if ipd.index is not None:
                        if hfn == None:
                            hfn = ipd.index
                        elif hfn != ipd.index and hfn is not True:
                            hfn = True
                            has_frame_numbers[column_name] = True
                            if hs is True:
                                break
                    if ipd.series is not None:
                        if hs == None:
                            hs = ipd.series
                        elif hs != ipd.series and hs is not True:
                            hs = True
                            has_series[column_name] = True
                            if hfn is True:
                                break
                else:
                    continue
                break
            if has_frame_numbers[column_name]:
                self.table.insert_column(idx, f_frame % column_name)
                idx += 1
                column_counts[column_name] += 1
            if has_series[column_name]:
                self.table.insert_column(idx, f_series % column_name)
                idx += 1
                column_counts[column_name] += 1
                
        data = []
        errors = []
        for i, (keys, image_set) in enumerate(self.image_sets):
            row = [unicode(key) for key in keys]
            for column_name in self.column_names:
                ipds = image_set.get(column_name, [])
                if len(ipds) == 0:
                    row += ["-- No image! --"] * column_counts[column_name]
                    errors.append((i, column_name))
                elif len(ipds) > 1:
                    row.append("-- Multiple images! --\n" + 
                               "\n".join([ipd.path for ipd in ipds]))
                    row += ["-- Multiple images! --"] * (
                        column_counts[column_name] - 1)
                    errors.append((i, column_name))
                else:
                    ipd = ipds[0]
                    row += os.path.split(ipd.path)
                    if has_frame_numbers[column_name]:
                        row += [str(ipd.index)]
                    if has_series[column_name]:
                        row += [str(ipd.series)]
            data.append(row)
        self.table.data = data
        for error_row, column_name in errors:
            for f in (f_pathname, f_filename):
                self.table.set_cell_attribute(error_row, f % column_name, 
                                              self.table.ATTR_ERROR)
                self.table.set_row_attribute(error_row, self.table.ATTR_ERROR)

class MetadataPredicate(cps.Filter.FilterPredicate):
    '''A predicate that compares an ifd against a metadata key and value'''
    
    SYMBOL = "metadata"
    def __init__(self, display_name, display_fmt = "%s", **kwargs):
        subpredicates = [cps.Filter.DoesPredicate([]),
                         cps.Filter.DoesNotPredicate([])]
        
        super(self.__class__, self).__init__(
            self.SYMBOL, display_name, MetadataPredicate.do_filter, 
            subpredicates, **kwargs)
        self.display_fmt = display_fmt
        
    def set_metadata_keys(self, keys):
        '''Define the possible metadata keys to be matched against literal values
        
        keys - a list of keys
        '''
        sub_subpredicates = [
            cps.Filter.FilterPredicate(
                key, 
                self.display_fmt % key, 
                lambda ipd, match, key=key: 
                ipd.metadata.has_key(key) and
                ipd.metadata[key] == match,
                [cps.Filter.LITERAL_PREDICATE])
            for key in keys]
        #
        # The subpredicates are "Does" and "Does not", so we add one level
        # below that.
        #
        for subpredicate in self.subpredicates:
            subpredicate.subpredicates = sub_subpredicates
        
    @classmethod
    def do_filter(cls, arg, *vargs):
        '''Perform the metadata predicate's filter function
        
        The metadata predicate has subpredicates that look up their
        metadata key in the ipd and compare it against a literal.
        '''
        node_type, modpath, resolver = arg
        ipd = resolver.get_image_plane_details(modpath)
        return vargs[0](ipd, *vargs[1:])
    
    def test_valid(self, pipeline, *args):
        modpath = ["imaging","image.png"]
        ipd = cpp.ImagePlaneDetails("/imaging/image.png", None, None, None)
        self((cps.FileCollectionDisplay.NODE_IMAGE_PLANE, modpath,
              NamesAndTypes.FakeModpathResolver(modpath, ipd)), *args)


class ColorImageProvider(LoadImagesImageProviderURL):
    '''Provide a color image, tripling a monochrome plane if needed'''
    def __init__(self, name, url, series, index):
        LoadImagesImageProviderURL.__init__(self, name, url,
                                            rescale = True,
                                            series = series,
                                            index = index)
        
    def provide_image(self, image_set):
        image = LoadImagesImageProviderURL.provide_image(self, image_set)
        if image.pixel_data.ndim == 2:
            image.pixel_data = np.dstack([image.pixel_data] * 3)
        return image
    
class MonochromeImageProvider(LoadImagesImageProviderURL):
    '''Provide a monochrome image, combining RGB if needed'''
    def __init__(self, name, url, series, index, channel, rescale = True):
        LoadImagesImageProviderURL.__init__(self, name, url,
                                            rescale = rescale,
                                            series = series,
                                            index = index,
                                            channel = channel)
        
    def provide_image(self, image_set):
        image = LoadImagesImageProviderURL.provide_image(self, image_set)
        if image.pixel_data.ndim == 3:
            image.pixel_data = \
                np.sum(image.pixel_data, 2) / image.pixel_data.shape[2]
        return image
    
class MaskImageProvider(MonochromeImageProvider):
    '''Provide a boolean image, converting nonzero to True, zero to False if needed'''
    def __init__(self, name, url, series, index, channel):
        MonochromeImageProvider.__init__(self, name, url,
                                            rescale = True,
                                            series = series,
                                            index = index,
                                            channel = channel)
        
    def provide_image(self, image_set):
        image = MonochromeImageProvider.provide_image(self, image_set)
        if image.pixel_data.dtype.kind != 'b':
            image.pixel_data = image.pixel_data != 0
        return image
    
class ObjectsImageProvider(LoadImagesImageProviderURL):
    '''Provide a multi-plane integer image, interpreting an image file as objects'''
    def __init__(self, name, url, series, index):
        LoadImagesImageProviderURL.__init__(self, name, url,
                                            rescale = False,
                                            series = series,
                                            index = index)
    def provide_image(self, image_set):
        """Load an image from a pathname
        """
        self.cache_file()
        filename = self.get_filename()
        channel_names = []
        url = self.get_url()
        properties = {}
        if self.series is not None:
            properties["series"] = self.series
        if self.index is not None:
            indexes = [self.index]
        else:
            metadata = get_omexml_metadata(self.get_full_name())
                                           
            ometadata = OME.OMEXML(metadata)
            pixel_metadata = ometadata.image(0 if self.series is None
                                             else self.series).Pixels
            nplanes = (pixel_metadata.SizeC * pixel_metadata.SizeZ * 
                       pixel_metadata.SizeT)
            indexes = range(nplanes)
            
        planes = []
        offset = 0
        for index in indexes:
            properties["index"] = str(index)
            img = load_using_bioformats(
                self.get_full_name(), 
                rescale=False, **properties).astype(int)
            img = convert_image_to_objects(img).astype(np.int32)
            img[img != 0] += offset
            offset += np.max(img)
            planes.append(img)
            
        image = cpi.Image(np.dstack(planes),
                          path_name = self.get_pathname(),
                          file_name = self.get_filename(),
                          convert=False)
        return image
    