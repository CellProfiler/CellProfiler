'''<b>Images</b> helps you collect the image files for your pipeline.

<hr>
TO DO: document this
'''

import cellprofiler.cpmodule as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.utilities.walk_in_background as W
import os
import urllib

class Images(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Images"
    category = "File Processing"
    
    
    def create_settings(self):
        self.walk_collection = W.WalkCollection(self.on_walk_completed)
        self.pipeline = None
        self.file_collection_display = cps.FileCollectionDisplay(
            "", "", self.on_drop, self.on_remove, 
            self.get_path_info, self.on_menu_command,
            self.handle_walk_pause_resume_stop)
        predicates = [FilePredicate(),
                      DirectoryPredicate(),
                      ExtensionPredicate(),
                      ImagePredicate()]
        self.wants_filter = cps.Binary(
            "Filter based on rules", False,
            doc = "Check this setting to display and use the rules filter")
            
        self.filter = cps.Filter("Filter", predicates, 
                                 'or (file does contain "")')
        self.do_filter = cps.DoSomething("","Apply filter", 
                                         self.apply_filter)
    
    def on_walk_completed(self):
        if self.pipeline is not None:
            self.file_collection_display.update_ui()
            
    def on_activated(self, pipeline):
        '''The module is gui-activated
        
        pipeline - the module's pipeline
        '''
        assert isinstance(pipeline, cpp.Pipeline)
        self.initialize_file_collection_display(pipeline)
        self.pipeline = pipeline
        pipeline.add_listener(self.pipeline_listener)
        
    def on_deactivated(self):
        '''The module is no longer in the GUI'''
        if self.pipeline is not None:
            self.pipeline.remove_listener(self.pipeline_listener)
            self.pipeline = None
        
    def on_drop(self, pathnames):
        '''Called when the UI is asking to add files or directories'''
        if self.pipeline is not None:
            self.pipeline.walk_paths(pathnames)
            
    def on_remove(self, mods):
        '''Called when the UI is asking to remove files from the list
        
        mods - two-tuple modpaths (see docs for FileCollectionDisplay)
        '''
        if self.pipeline is not None:
            self.pipeline.start_undoable_action()
            try:
                ipds = self.make_ipds_from_mods(mods)
                self.pipeline.remove_image_plane_details(ipds)
            finally:
                self.pipeline.stop_undoable_action()
            
    def get_path_info(self, modpath):
        '''Get a descriptive name, the image type, the tooltip and menu for a path'''
        exemplar = self.make_ipd_from_modpath(modpath)
        if self.pipeline is None:
            ipd = None
        else:
            ipd = self.pipeline.find_image_plane_details(exemplar)
        if ipd is None:
            return exemplar.path, self.file_collection_display.NODE_FILE, None, []
        size_t = int(ipd.metadata.get(cpp.ImagePlaneDetails.MD_SIZE_T, 1))
        size_z = int(ipd.metadata.get(cpp.ImagePlaneDetails.MD_SIZE_Z, 1))
        size_c = int(ipd.metadata.get(cpp.ImagePlaneDetails.MD_SIZE_C, 1))
        color_format = ipd.metadata.get(cpp.ImagePlaneDetails.MD_COLOR_FORMAT,
                                        None)
        if color_format is None:
            image_type = self.file_collection_display.NODE_FILE
        elif size_t > 1 and size_z == 1:
            image_type = self.file_collection_display.NODE_MOVIE
        elif (size_z > 1 or 
              (color_format == cpp.ImagePlaneDetails.MD_PLANAR and size_c > 1)):
            image_type = self.file_collection_display.NODE_COMPOSITE_IMAGE
        elif color_format == cpp.ImagePlaneDetails.MD_RGB:
            image_type = self.file_collection_display.NODE_COLOR_IMAGE
        else:
            if isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
                image_type = self.file_collection_display.NODE_IMAGE_PLANE
            else:
                image_type = self.file_collection_display.NODE_MONOCHROME_IMAGE
        name = os.path.split(ipd.path)[1]
        if isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
            series, index, channel = modpath[-1]
            if (ipd.metadata.has_key(cpp.ImagePlaneDetails.MD_CHANNEL_NAME)):
                name = "%s (series =%2d, index =%3d, channel=%s)" % (
                    name, series, index, 
                    ipd.metadata[cpp.ImagePlaneDetails.MD_CHANNEL_NAME])
            else:
                name = "%s (series =%2d, index=%3d)" % (name, series, index)
            
        return name, image_type, None, []
    
    def on_menu_command(self, command):
        pass
    
    def handle_walk_pause_resume_stop(self, command):
        if self.pipeline is not None:
            if command == self.file_collection_display.BKGND_PAUSE:
                self.pipeline.file_walker.pause()
            elif command == self.file_collection_display.BKGND_RESUME:
                self.pipeline.file_walker.resume()
            elif command == self.file_collection_display.BKGND_STOP:
                self.pipeline.file_walker.stop()
            elif self.pipeline.file_walker.get_state() == W.THREAD_PAUSE:
                return self.file_collection_display.BKGND_PAUSE
            elif (self.pipeline.file_walker.get_state() in 
                  (W.THREAD_STOP, W.THREAD_STOPPING)):
                return self.file_collection_display.BKGND_STOP
            else:
                return self.file_collection_display.BKGND_RESUME
        
    def pipeline_listener(self, pipeline, event):
        if isinstance(event, cpp.ImagePlaneDetailsAddedEvent):
            self.on_ipds_added(event.image_plane_details)
        elif isinstance(event, cpp.ImagePlaneDetailsMetadataEvent):
            self.on_ipd_metadata_change(event.image_plane_details)
        elif isinstance(event, cpp.ImagePlaneDetailsRemovedEvent):
            self.on_ipds_removed(event.image_plane_details)
        elif isinstance(event, cpp.FileWalkStartedEvent):
            self.file_collection_display.update_ui(
                cps.FileCollectionDisplay.BKGND_RESUME)
        elif isinstance(event, cpp.FileWalkEndedEvent):
            self.file_collection_display.update_ui(
                cps.FileCollectionDisplay.BKGND_STOP)
            
    def on_ipds_added(self, image_plane_details):
        modlist = []
        for ipd in image_plane_details:
            self.add_ipd_to_modlist(ipd, modlist)
        self.file_collection_display.add(modlist)
        
    def on_ipds_removed(self, image_plane_details):
        modlist = []
        for ipd in image_plane_details:
            self.add_ipd_to_modlist(ipd, modlist)
        self.file_collection_display.remove(modlist)
        
    def on_ipd_metadata_change(self, ipd):
            self.file_collection_display.modify(self.add_ipd_to_modlist(ipd))
            
    @classmethod
    def add_ipd_to_modlist(self, ipd, modlist = None):
        '''Add an image plane details to a modlist
        
        modlist - a list of two-tuples as described in cps.FileCollectionDisplay
        
        ipd - the image plane details record to add
        '''
        if modlist is None:
            modlist = []
        modpath = self.make_modpath_from_ipd(ipd)
        return self.add_modpath_to_modlist(modpath, modlist)
        
    @classmethod
    def add_modpath_to_modlist(self, modpath, modlist = None):
        if modlist is None:
            modlist = []
        if len(modpath) == 1:
            modlist.append(modpath[0])
            return modlist
        idxs = [i for i, mod in enumerate(modlist)
                if isinstance(mod, tuple) and len(mod) == 2 and
                mod[0] == modpath[0]]
        if len(idxs) == 0:
            modlist.append((modpath[0], self.add_modpath_to_modlist(modpath[1:])))
            return modlist
        idx = idxs[0]
        self.add_modpath_to_modlist(modpath[1:], modlist[idx][1])
        return modlist
        
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
                    if not (dd.has_key(part) and dd[part] is not None):
                        dd[part] = {}
                    dd = dd[part]
                dd[parts[0]] = None
        mods = self.make_mods_from_tree(d)
        self.file_collection_display.initialize_tree(mods)
        self.apply_filter()
        
    def make_mods_from_tree(self, tree):
        '''Create a list of FileCollectionDisplay mods from a tree'''
        return [k if tree[k] is None
                else (k, self.make_mods_from_tree(tree[k]))
                for k in tree.keys()]
    
    @classmethod
    def make_parts_list_from_mod(self, mod):
        '''Convert a mod to a collection of parts lists
        
        A mod has the form (root, [...]). Convert this recursively
        to get lists starting with "root" for each member of the list.
        '''
        if cps.FileCollectionDisplay.is_leaf(mod):
            return [[mod]]
        root, modlist = mod
        return sum([[[root]+ x for x in self.make_parts_list_from_mod(submod)]
                    for submod in modlist],[])
    
    def make_ipds_from_mods(self, mods):
        ipds = []
        for mod in mods:
            for parts_list in self.make_parts_list_from_mod(mod):
                ipds.append(self.make_ipd_from_modpath(parts_list))
        return ipds
    
    @classmethod
    def make_ipd_from_modpath(cls, modpath):
        if isinstance(modpath[-1], tuple):
            path = os.path.join(*modpath[:-1])
            series, index, channel = modpath[-1]
        else:
            path = os.path.join(*modpath)
            series = index = channel = None
        url = "file:" + urllib.pathname2url(path)
        return cpp.ImagePlaneDetails(url, series, index, channel)
    
    @classmethod
    def make_modpath_from_ipd(cls, ipd):
        path = ipd.path
        result = []
        while True:
            new_path, part = os.path.split(path)
            if len(new_path) == 0 or len(part) == 0:
                if all([x is None for x in (ipd.series, ipd.index, ipd.channel)]):
                    return [path] + result
                return [path] + result + [(ipd.series, ipd.index, ipd.channel)]
            result.insert(0, part)
            path = new_path
            
    def get_image_plane_details(self, modpath):
        '''Find the image plane details, given a path list
        
        modpath - the list of file parts, starting at the root
        
        series - the series of the image plane within the file
        
        index - the index of the image plane within the file
        
        channel - the channel within the file
        '''
        if self.pipeline is None:
            return None
        if cps.FileCollectionDisplay.mod_is_image_plane(modpath[-1]):
            series, index, channel = modpath[-1]
            modpath = modpath[:-1]
        else:
            series = index = channel = None
        path = os.path.join(*modpath)
        exemplar = cpp.ImagePlaneDetails("file:" + urllib.pathname2url(path),
                                         series, index, channel)
        return self.pipeline.find_image_plane_details(exemplar)
        
    def apply_filter(self):
        if not self.wants_filter:
            def all_mods(tree):
                result = []
                for key in tree.keys():
                    if key is None:
                        continue
                    if isinstance(tree[key], bool):
                        result.append(key)
                    else:
                        result.append((key, all_mods(tree[key])))
                return result
            keep = all_mods(self.file_collection_display.file_tree)
            self.file_collection_display.mark(keep, True)
        else:
            keep = []
            dont_keep = []
            self.filter_tree(self.file_collection_display.file_tree, keep, dont_keep)
            self.file_collection_display.mark(keep, True)
            self.file_collection_display.mark(dont_keep, False)
            
    def filter_ipd(self, ipd):
        '''Return True if an image plane descriptor should be kept'''
        if not self.wants_filter:
            return True
        modpath = self.make_modpath_from_ipd(ipd)
        match = self.filter.evaluate((
            cps.FileCollectionDisplay.NODE_IMAGE_PLANE, modpath, self))
        return match or match is None
    
    def filter_tree(self, tree, keep, dont_keep, modpath = []):
        for key in tree.keys():
            if key is None:
                continue
            subpath = modpath + [key]
            if isinstance(tree[key], bool):
                display_name, node_type, tooltip, menu = self.get_path_info(
                    subpath)
                match = self.filter.evaluate((node_type, subpath, self))
                if match is None or match:
                    keep.append(key)
                else:
                    dont_keep.append(key)
            else:
                keep_node = (key, [])
                keep.append(keep_node)
                dont_keep_node = (key, [])
                dont_keep.append(dont_keep_node)
                self.filter_tree(tree[key], keep_node[1], dont_keep_node[1], 
                                 subpath)
    
    def settings(self):
        return [self.file_collection_display, self.wants_filter, self.filter]
    
    def visible_settings(self):
        result = [self.file_collection_display, self.wants_filter]
        if self.wants_filter:
            result += [self.filter, self.do_filter]
        return result
            
    def on_setting_changed(self, setting, pipeline):
        if setting is self.wants_filter and not self.wants_filter:
            self.apply_filter()
    
    def run(self):
        pass
    
class DirectoryPredicate(cps.Filter.FilterPredicate):
    '''A predicate that only filters directories'''
    def __init__(self):
        subpredicates = (
            cps.Filter.CONTAINS_PREDICATE,
            cps.Filter.CONTAINS_REGEXP_PREDICATE,
            cps.Filter.STARTS_WITH_PREDICATE,
            cps.Filter.ENDSWITH_PREDICATE,
            cps.Filter.EQ_PREDICATE)
        predicates = [cps.Filter.DoesPredicate(subpredicates),
                      cps.Filter.DoesNotPredicate(subpredicates)]
        cps.Filter.FilterPredicate.__init__(self,
            'directory', "Directory", self.fn_filter,
            predicates, doc = "Apply the rule to directories")
        
    def fn_filter(self, (node_type, modpath, module), *args):
        '''The DirectoryPredicate filter function
        
        The arg slot expects a tuple of node_type and modpath.
        The predicate returns None (= agnostic about filtering) if
        the node is not a directory, otherwise it composites the
        modpath into a file path and applies it to the rest of
        the args.
        '''
        if isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
            path = os.path.join(*modpath[:-2])
        else:
            path = os.path.join(*modpath[:-1])
        return args[0](path, *args[1:])
    
    def test_valid(self, pipeline, *args):
        self((cps.FileCollectionDisplay.NODE_FILE, 
              ["/imaging","image.tif"], None), *args)

class FilePredicate(cps.Filter.FilterPredicate):
    '''A predicate that only filters files'''
    def __init__(self):
        subpredicates = (
            cps.Filter.CONTAINS_PREDICATE,
            cps.Filter.CONTAINS_REGEXP_PREDICATE,
            cps.Filter.STARTS_WITH_PREDICATE,
            cps.Filter.ENDSWITH_PREDICATE,
            cps.Filter.EQ_PREDICATE)     
        predicates = [cps.Filter.DoesPredicate(subpredicates),
                      cps.Filter.DoesNotPredicate(subpredicates)]
        cps.Filter.FilterPredicate.__init__(self,
            'file', "File", self.fn_filter, predicates,
            doc = "Apply the rule to files")
        
    def fn_filter(self, (node_type, modpath, module), *args):
        '''The FilePredicate filter function
        
        The arg slot expects a tuple of node_type and modpath.
        The predicate returns None (= agnostic about filtering) if
        the node is not a directory, otherwise it composites the
        modpath into a file path and applies it to the rest of
        the args
        '''
        if node_type == cps.FileCollectionDisplay.NODE_DIRECTORY:
            return None
        elif isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
            filename = modpath[-2]
        else:
            filename = modpath[-1]
        return args[0](filename, *args[1:])
    
    def test_valid(self, pipeline, *args):
        self((cps.FileCollectionDisplay.NODE_FILE, 
              ["/imaging", "test.tif"], None), *args)

class ExtensionPredicate(cps.Filter.FilterPredicate):
    '''A predicate that operates on file extensions'''
    IS_TIF_PREDICATE = cps.Filter.FilterPredicate(
        "istif", '"tif", "tiff", "ome.tif" or "ome.tiff"',
        lambda x: x.lower() in ("tif", "tiff", "ome.tif", "ome.tiff"), [],
        doc="The extension is associated with TIFF image files")
    IS_JPEG_PREDICATE = cps.Filter.FilterPredicate(
        "isjpeg", '"jpg" or "jpeg"',
        lambda x: x.lower() in ("jpg", "jpeg"), [],
        doc = "The extension is associated with JPEG image files")
    IS_PNG_PREDICATE = cps.Filter.FilterPredicate(
        "ispng", '"png"',
        lambda x: x.lower() == "png", [],
        doc = "The extension is associated with PNG image files")
    IS_IMAGE_PREDICATE = cps.Filter.FilterPredicate(
        'isimage', 'the extension of an image file',
        lambda x: any([Images.ExtensionPredicate.IS_TIF_PREDICATE(x), 
                       Images.ExtensionPredicate.IS_JPEG_PREDICATE(x),
                       Images.ExtensionPredicate.IS_PNG_PREDICATE(x)]), [],
        'Is an extension commonly associated with image files')
    IS_FLEX_PREDICATE = cps.Filter.FilterPredicate(
        'isflex', '"flex"',
        lambda x: x.lower() == "flex", [],
        doc = "The extension is associated with .flex files")
    IS_MOVIE_PREDICATE = cps.Filter.FilterPredicate(
        "ismovie", '"mov" or "avi"',
        lambda x: x.lower() in ("mov", "avi"), [],
        doc = "The extension is associated with movie files")
    def __init__(self):
        subpredicates = (
            self.IS_TIF_PREDICATE,
            self.IS_JPEG_PREDICATE,
            self.IS_PNG_PREDICATE,
            self.IS_IMAGE_PREDICATE,
            self.IS_FLEX_PREDICATE,
            self.IS_MOVIE_PREDICATE)            
        predicates = [ cps.Filter.DoesPredicate(subpredicates, "Is"),
                       cps.Filter.DoesNotPredicate(subpredicates, "Is not")]
        cps.Filter.FilterPredicate.__init__(self,
            'extension', "Extension", self.fn_filter, predicates,
            doc="The rule applies to the file extension")
        
    def fn_filter(self, (node_type, modpath, module), *args):
        '''The ExtensionPredicate filter function
        
        If the element is a file, try the different predicates on 
        all possible extension parsings.
        '''
        if node_type == cps.FileCollectionDisplay.NODE_DIRECTORY:
            return None
        elif isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
            filename = modpath[-2]
        else:
                filename = modpath[-1]
        exts = []
        while True:
            filename, ext = os.path.splitext(filename)
            if len(filename) == 0 or len(ext) == 0:
                return False
            exts.insert(0, ext[1:])
            ext = '.'.join(exts)
            if args[0](ext, *args[1:]):
                return True
            
    def test_valid(self, pipeline, *args):
        self((cps.FileCollectionDisplay.NODE_FILE, 
              ["/imaging", "test.tif"], None), *args)
            
class ImagePredicate(cps.Filter.FilterPredicate):
    '''A predicate that applies subpredicates to image plane details'''
    IS_COLOR_PREDICATE = cps.Filter.FilterPredicate(
        "iscolor", "Color", 
        lambda x: (
            x.metadata.has_key(cpp.ImagePlaneDetails.MD_COLOR_FORMAT) and
            x.metadata[cpp.ImagePlaneDetails.MD_COLOR_FORMAT] == 
            cpp.ImagePlaneDetails.MD_RGB), [],
        doc = "The image is an interleaved color image (for example, a PNG image)")
    
    IS_MONOCHROME_PREDICATE = cps.Filter.FilterPredicate(
        "ismonochrome", "Monochrome", 
        lambda x: (
            x.metadata.has_key(cpp.ImagePlaneDetails.MD_COLOR_FORMAT) and
            x.metadata[cpp.ImagePlaneDetails.MD_COLOR_FORMAT] ==
            cpp.ImagePlaneDetails.MD_MONOCHROME), [],
        doc = "The image is monochrome")
    
    def __init__(self):
        subpredicates = ( self.IS_COLOR_PREDICATE, 
                          self.IS_MONOCHROME_PREDICATE)
        predicates = [ pred_class(subpredicates, text)
                       for pred_class, text in (
                           (cps.Filter.DoesPredicate, "Is"),
                           (cps.Filter.DoesNotPredicate, "Is not"))]
        cps.Filter.FilterPredicate.__init__(self,
            'image', "Image", self.fn_filter,
            predicates,
            doc = "Filter based on image characteristics")
        
    def fn_filter(self, (node_type, modpath, module), *args):
        if node_type == cps.FileCollectionDisplay.NODE_DIRECTORY:
            return None
        ipd = module.get_image_plane_details(modpath)
        if ipd is None:
            return None
        return args[0](ipd, *args[1:])

    def test_valid(self, pipeline, *args):
        image_module = [m for m in pipeline.modules()
                        if isinstance(m, Images)][0]
        self((cps.FileCollectionDisplay.NODE_FILE, 
              ["/imaging", "test.tif"], image_module), *args)

