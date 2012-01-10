'''<b>Images</b> helps you collect the image files for your pipeline.

<hr>
TO DO: document this
'''

import cellprofiler.cpmodule as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import os
import urllib

class Images(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Images"
    category = "File Processing"

    def create_settings(self):
        self.file_collection_display = cps.FileCollectionDisplay(
            "", "", self.on_fcd_change, self.get_image_plane_details)
        subpredicates = [cps.Filter.CONTAINS_PREDICATE,
                         cps.Filter.CONTAINS_REGEXP_PREDICATE,
                         cps.Filter.STARTS_WITH_PREDICATE,
                         cps.Filter.ENDSWITH_PREDICATE,
                         cps.Filter.EQ_PREDICATE]
        predicates = [cls(subpredicates) for cls in 
                      (cps.Filter.DoesPredicate, cps.Filter.DoesNotPredicate)]
        self.filter = cps.Filter("Filter", predicates, 'or (doesnot contain "foo") (and (does startwith "foo") (does endwith "bar"))')
        self.pipeline = None
        self.modifying_ipds = False
    
    def on_activated(self, pipeline):
        '''The module is gui-activated
        
        pipeline - the module's pipeline
        '''
        assert isinstance(pipeline, cpp.Pipeline)
        self.initialize_file_collection_display(pipeline)
        self.pipeline = pipeline
        self.modifying_ipds = False
        pipeline.add_listener(self.pipeline_listener)
        
    def on_deactivated(self):
        '''The module is no longer in the GUI'''
        self.pipeline.remove_listener(self.pipeline_listener)
        self.pipeline = None
        
    def pipeline_listener(self, pipeline, event):
        if ((not self.modifying_ipds) and
            isinstance(event, (cpp.ImagePlaneDetailsAddedEvent, 
                               cpp.ImagePlaneDetailsRemovedEvent))):
            self.initialize_file_collection_display(self.pipeline)
        
    def initialize_file_collection_display(self, pipeline):
        d = {}
        for detail in pipeline.image_plane_details:
            assert isinstance(detail, cpp.ImagePlaneDetails)
            if detail.url.lower().startswith("file:"):
                path = urllib.url2pathname(detail.url[5:])
                parts = []
                while True:
                    new_path, part = os.path.split(path)
                    if len(new_path) == 0 or len(part) == 0:
                        parts.append(path)
                        break
                    parts.append(part)
                    path = new_path
                if (detail.series is not None or
                    detail.index is not None or
                    detail.channel is not None):
                    parts.insert(0, (detail.series, detail.index, detail.channel))
                dd = d
                for part in reversed(parts[1:]):
                    if not dd.has_key(part):
                        dd[part] = {}
                    dd = dd[part]
                dd[parts[0]] = None
        mods = self.make_mods_from_tree(d)
        self.file_collection_display.initialize_tree(mods)
        self.on_filter_change()
        
    def make_mods_from_tree(self, tree):
        '''Create a list of FileCollectionDisplay mods from a tree'''
        return [k if tree[k] is None
                else (k, self.make_mods_from_tree(tree[k]))
                for k in tree.keys()]
    
    def make_parts_list_from_mod(self, mod):
        '''Convert a mod to a collection of parts lists
        
        A mod has the form (root, [...]). Convert this recursively
        to get lists starting with "root" for each member of the list.
        '''
        if (cps.FileCollectionDisplay.mod_is_file(mod) or
            cps.FileCollectionDisplay.mod_is_image_plane(mod)):
            return [[mod]]
        root, modlist = mod
        return sum([[[root]+ x for x in self.make_parts_list_from_mod(submod)]
                    for submod in modlist],[])
    
    def make_ipds_from_mods(self, mods):
        ipds = []
        for mod in mods:
            for parts_list in self.make_parts_list_from_mod(mod):
                if cps.FileCollectionDisplay.mod_is_image_plane(parts_list[-1]):
                    path = os.path.join(*parts_list[:-1])
                    series, index, channel = parts_list[-1]
                else:
                    path = os.path.join(*parts_list)
                    series = index = channel = None
                url = "file:" + urllib.pathname2url(path)
                ipds.append(cpp.ImagePlaneDetails(url, series, index, channel))
        return ipds
        
    def on_fcd_change(self, operation, *args):
        if self.pipeline:
            self.modifying_ipds = True
            try:
                if operation in (cps.FileCollectionDisplay.ADD,
                                 cps.FileCollectionDisplay.REMOVE):
                    mods = args[0]
                    ipds = self.make_ipds_from_mods(mods)
                if operation == cps.FileCollectionDisplay.ADD:
                    self.pipeline.add_image_plane_details(ipds)
                elif operation == cps.FileCollectionDisplay.REMOVE:
                    self.pipeline.remove_image_plane_details(ipds)
                elif operation == cps.FileCollectionDisplay.METADATA:
                    path, metadata = args
                    ipd = self.get_image_plane_details(path)
                    if ipd is not None:
                        ipd.metadata.update(metadata)
            finally:
                self.modifying_ipds = False
                
    def get_image_plane_details(self, modpath):
        '''Find the image plane details, given a path list
        
        modpath - the list of file parts, starting at the root
        
        series - the series of the image plane within the file
        
        index - the index of the image plane within the file
        
        channel - the channel within the file
        '''
        if cps.FileCollectionDisplay.mod_is_image_plane(modpath[-1]):
            series, index, channel = modpath[-1]
            modpath = modpath[:-1]
        else:
            series = index = channel = None
        path = os.path.join(*modpath)
        exemplar = cpp.ImagePlaneDetails("file:" + urllib.pathname2url(path),
                                         series, index, channel)
        return self.pipeline.find_image_plane_details(exemplar)
        
    def on_filter_change(self):
        keep = []
        dont_keep = []
        self.filter_tree(self.file_collection_display.file_tree, keep, dont_keep)
        self.file_collection_display.mark(keep, True)
        self.file_collection_display.mark(dont_keep, False)
        
    def filter_tree(self, tree, keep, dont_keep):
        for key in tree.keys():
            if key is None:
                continue
            if isinstance(tree[key], bool):
                match = self.filter.evaluate(key)
                if match:
                    keep.append(key)
                else:
                    dont_keep.append(key)
            else:
                keep_node = (key, [])
                keep.append(keep_node)
                dont_keep_node = (key, [])
                dont_keep.append(dont_keep_node)
                self.filter_tree(tree[key], keep_node[1], dont_keep_node[1])
    
    def settings(self):
        return [self.file_collection_display, self.filter]
    
    def run(self):
        pass
    
    def on_setting_changed(self, setting, pipeline):
        if setting is self.filter:
            self.on_filter_change()
    