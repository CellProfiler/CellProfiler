'''<b>Images</b> helps you collect the image files for your pipeline.

<hr>
TO DO: document this
'''

import cellprofiler.cpmodule as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import os
import urllib

NODE_DIRECTORY = "DirectoryNode"
NODE_FILE = "FileNode"
NODE_IMAGE_PLANE = "ImagePlaneNode"

class Images(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Images"
    category = "File Processing"
    
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
            if node_type == NODE_DIRECTORY:
                path = os.path.join(*modpath)
            elif node_type == NODE_FILE:
                path = os.path.join(*modpath[:-1])
            elif node_type == NODE_IMAGE_PLANE:
                path = os.path.join(*modpath[:-2])
            return args[0](path, *args[1:])
        
        def test_valid(self, pipeline, *args):
            self((NODE_FILE, ["/imaging","image.tif"], None), *args)

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
            if node_type == NODE_FILE:
                filename = modpath[-1]
            elif node_type == NODE_IMAGE_PLANE:
                filename = modpath[-2]
            else:
                return None
            return args[0](filename, *args[1:])
        
        def test_valid(self, pipeline, *args):
            self((NODE_FILE, ["/imaging", "test.tif"], None), *args)

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
            if node_type != NODE_FILE:
                return None
            exts = []
            filename = modpath[-1]
            while True:
                filename, ext = os.path.splitext(filename)
                if len(filename) == 0 or len(ext) == 0:
                    return False
                exts.insert(0, ext[1:])
                ext = '.'.join(exts)
                if args[0](ext, *args[1:]):
                    return True
                
        def test_valid(self, pipeline, *args):
            self((NODE_FILE, ["/imaging", "test.tif"], None), *args)
                
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
            if node_type not in (NODE_IMAGE_PLANE, NODE_FILE):
                return None
            ipd = module.get_image_plane_details(modpath)
            if ipd is None:
                return None
            return args[0](ipd, *args[1:])

        def test_valid(self, pipeline, *args):
            image_module = [m for m in pipeline.modules()
                            if isinstance(m, Images)][0]
            self((NODE_FILE, ["/imaging", "test.tif"], image_module), *args)

    def create_settings(self):
        self.file_collection_display = cps.FileCollectionDisplay(
            "", "", self.on_fcd_change, self.get_image_plane_details)
        predicates = [self.DirectoryPredicate(),
                      self.FilePredicate(),
                      self.ExtensionPredicate(),
                      self.ImagePredicate()]
        self.filter = cps.Filter("Filter", predicates, 
                                 'or (file does contain "")')
        self.do_filter = cps.DoSomething("","Apply filter", 
                                         self.apply_filter)
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
        self.apply_filter()
        
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
        keep = []
        dont_keep = []
        self.filter_tree(self.file_collection_display.file_tree, keep, dont_keep)
        self.file_collection_display.mark(keep, True)
        self.file_collection_display.mark(dont_keep, False)
    
    def filter_tree(self, tree, keep, dont_keep, modpath = []):
        for key in tree.keys():
            if key is None:
                continue
            subpath = modpath + [key]
            if isinstance(tree[key], bool):
                if cps.FileCollectionDisplay.mod_is_image_plane(key):
                    node_type = NODE_IMAGE_PLANE
                else:
                    node_type = NODE_FILE
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
        return [self.file_collection_display, self.filter]
    
    def visible_settings(self):
        return [self.file_collection_display, self.filter, self.do_filter]
    
    def run(self):
        pass
    