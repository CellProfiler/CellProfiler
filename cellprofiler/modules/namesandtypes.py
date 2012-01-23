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
import cellprofiler.cpmodule as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
from cellprofiler.modules.images import FilePredicate
from cellprofiler.modules.images import ExtensionPredicate
from cellprofiler.modules.images import ImagePredicate
from cellprofiler.modules.images import DirectoryPredicate
from cellprofiler.modules.images import Images, NODE_IMAGE_PLANE

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

IDX_ASSIGNMENTS_COUNT = 4

class NamesAndTypes(cpm.CPModule):
    variable_revision_number = 1
    module_name = "NamesAndTypes"
    category = "File Processing"
    
    def create_settings(self):
        self.pipeline = None
        self.ipds = []
        self.image_sets = []
        
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
            "Image name", "DNA")
            
        self.assignments = []
        self.assignments_count = cps.HiddenCount( self.assignments,
                                                  "Assignments count")
        self.add_assignment(can_remove = False)
        self.add_assignment_button = cps.DoSomething(
            "Add another assignment rule", "Add",
            self.add_assignment)
        self.join = cps.Joiner("")
        self.table = cps.Table("")
        
    def add_assignment(self, can_remove = True):
        '''Add a rules assignment'''
        group = cps.SettingsGroup()
        self.assignments.append(group)
        
        if can_remove:
            group.append("divider", cps.Divider())
        
        group.append("rule_filter", cps.Filter(
            "Match this rule",
            [FilePredicate(),
             DirectoryPredicate(),
             ExtensionPredicate(),
             ImagePredicate(),
             MetadataPredicate("Metadata", "Have %s matching", 
                               doc="Has metadata matching the value you enter")],
            'or (file does contain "")'))
        
        group.append("image_name", cps.ImageNameProvider(
            "Image name", "DNA"))
        
        group.append("object_name", cps.ObjectNameProvider(
            "Objects name", "Cells"))
        
        group.append("load_as_choice", cps.Choice(
            "Load as", LOAD_AS_ALL))
        
        group.can_remove = can_remove
        if can_remove:
            group.append("remover", cps.RemoveSettingButton(
                'Remove above rule', "Remove", 
                self.assignments, group))
            
    def settings(self):
        result = [self.assignment_method, self.single_load_as_choice,
                  self.single_image_provider, self.join, self.assignments_count]
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
                result += [self.join]
        result += [self.table]
        return result
    
    def prepare_settings(self, setting_values):
        n_assignments = int(setting_values[IDX_ASSIGNMENTS_COUNT])
        if len(self.assignments) > n_assignments:
            del self.assignments[n_assignments:]
        while len(self.assignments) < n_assignments:
            self.add_assignment()
            
    def run(self, workspace):
        pass
                     
    def on_activated(self, pipeline):
        self.pipeline = pipeline
        self.ipds = pipeline.get_filtered_image_plane_details(with_metadata=True)
        self.metadata_keys = set()
        for ipd in self.ipds:
            self.metadata_keys.update(ipd.metadata.keys())
        self.update_all_metadata_predicates()
        self.update_all_columns()
        
    def on_deactivated(self):
        self.pipeline = None
        
    def on_setting_changed(self, setting, pipeline):
        '''Handle updates to the join control'''
        if setting == self.assignment_method:
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
                            self.make_image_sets()
                            self.update_table()
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
        self.make_image_sets()
        self.update_table()
        self.update_joiner()
        
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
            self.image_sets = [((str(i+1), ), { column_name: (ipd,) })
                               for i, ipd in enumerate(self.ipd_columns[0])]
        elif self.assignment_method == ASSIGN_RULES:
            #
            # d is a nested dictionary. The key is either None (= match all)
            # or the first metadata value. The value is either a subdictionary
            # or the leaf which is an image_set dictionary.
            #
            if len(self.column_names) == 1:
                column_name = self.column_names[0]
                self.image_sets = [((str(i+1), ), { column_name: (ipd,) })
                                   for i, ipd in enumerate(self.ipd_columns[0])]
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
        modpath = Images.make_modpath_from_ipd(ipd)
        try:
            match = group.rule_filter.evaluate(
                (NODE_IMAGE_PLANE, modpath, self.FakeModpathResolver(modpath, ipd)))
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
                        for i, column_name in enumerate(column_names):
                            if not join.has_key(column_name):
                                if best_value in self.column_metadata_choices[i]:
                                    join[column_name] = best_value
                                else:
                                    join[column_name] = None
            except:
                pass # bad field value
    
    def update_table(self):
        '''Update the table to show the current image sets'''
        joins = self.join.parse()
        self.table.clear_columns()
        self.table.clear_rows()
        
        if (self.assignment_method == ASSIGN_RULES and 
            len(self.column_names) > 1):
            metadata_columns = [
                " / ".join(set([k for k in join.values() if k is not None]))
                for join in joins]
        else:
            metadata_columns = ["Image number"]
        for i, name in enumerate(metadata_columns + self.column_names):
            self.table.insert_column(i, name)
        data = []
        for keys, image_set in self.image_sets:
            row = list(keys)
            for column_name in self.column_names:
                ipds = image_set[column_name]
                if len(ipds) == 0:
                    row.append("-- No image! --")
                elif len(ipds) > 1:
                    row.append("-- Multiple images! --\n" + 
                               "\n".join([ipd.path for ipd in ipds]))
                else:
                    row.append(ipds[0].path)
            data.append(row)
        self.table.data = data

class MetadataPredicate(cps.Filter.FilterPredicate):
    '''A predicate that compares an ifd against a metadata key and value'''
    
    def __init__(self, display_name, display_fmt = "%s", **kwargs):
        super(self.__class__, self).__init__(
            "metadata", display_name, MetadataPredicate.do_filter, [], **kwargs)
        self.display_fmt = display_fmt
        
    def set_metadata_keys(self, keys):
        '''Define the possible metadata keys to be matched against literal values
        
        keys - a list of keys
        '''
        self.subpredicates = [
            cps.Filter.FilterPredicate(
                key, self.display_fmt % key, 
                lambda ipd, match: 
                ipd.metadata.has_key(self.symbol) and
                ipd.metadata[self.symbol] == match,
                [cps.Filter.LITERAL_PREDICATE])
            for key in keys]
        
    @classmethod
    def do_filter(cls, arg, matcher, literal):
        '''Perform the metadata predicate's filter function
        
        The metadata predicate has subpredicates that look up their
        metadata key in the ipd and compare it against a literal.
        '''
        node_type, modpath, resolver = arg
        ipd = resolver.get_image_plane_details(modpath)
        return matcher(ipd, literal)
    
    def test_valid(self, pipeline, *args):
        modpath = ["imaging","image.png"]
        ipd = cpp.ImagePlaneDetails("/imaging/image.png", None, None, None)
        self((NODE_IMAGE_PLANE, modpath,
              NamesAndTypes.FakeModpathResolver(modpath, ipd)), *args)
