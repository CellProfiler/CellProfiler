"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

'''<b>Images</b> helps you collect the image files for your pipeline.

<hr>
TO DO: document this
'''

import cellprofiler.cpmodule as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw
import cellprofiler.utilities.walk_in_background as W
import os
import sys
import urllib

from .loadimages import pathname2url, SUPPORTED_IMAGE_EXTENSIONS
from .loadimages import SUPPORTED_MOVIE_EXTENSIONS

class Images(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Images"
    category = "File Processing"
    
    MI_SHOW_IMAGE = "Show image"
    MI_REMOVE = cps.FileCollectionDisplay.DeleteMenuItem("Remove from list")
    MI_REFRESH = "Refresh"
    
    def create_settings(self):
        self.workspace = None
        self.file_collection_display = cps.FileCollectionDisplay(
            "", "", self.on_drop, self.on_remove, 
            self.get_path_info, self.on_menu_command,
            self.handle_walk_pause_resume_stop)
        predicates = [FilePredicate(),
                      DirectoryPredicate(),
                      ExtensionPredicate()]
        self.wants_filter = cps.Binary(
            "Filter based on rules", False,
            doc = "Check this setting to display and use the rules filter")
            
        self.filter = cps.Filter("Filter", predicates, 
                                 'or (file does contain "")')
        self.do_filter = cps.DoSomething("","Apply filter", 
                                         self.apply_filter)
    
    def on_walk_completed(self):
        if self.workspace is not None:
            self.file_collection_display.update_ui(
                self.file_collection_display.BKGND_STOP)
            
    def on_walk_callback(self, dirpath, dirnames, filenames):
        '''Handle an iteration of file walking'''
        if self.workspace is not None:
            hdf_file_list = self.workspace.get_file_list()
            file_list = [pathname2url(os.path.join(dirpath, filename))
                         for filename in filenames]
            hdf_file_list.add_files_to_filelist(file_list)
            modpath = self.make_modpath_from_path(dirpath)
            modlist = self.add_files_to_modlist(modpath, filenames)
            self.file_collection_display.add(modlist)
            
    def on_activated(self, workspace):
        '''The module is gui-activated
        
        workspace - the user's workspace
        '''
        self.workspace = workspace
        assert isinstance(workspace, cpw.Workspace)
        self.initialize_file_collection_display(workspace)
        
    def on_deactivated(self):
        '''The module is no longer in the GUI'''
        self.workspace = None
        
    def on_drop(self, pathnames, check_for_directories):
        '''Called when the UI is asking to add files or directories'''
        if self.workspace is not None:
            modlist = []
            urls = []
            for pathname in pathnames:
                # Hack - convert drive names to lower case in
                #        Windows to normalize them.
                if (sys.platform == 'win32' and pathname[0].isalpha()
                    and pathname[1] == ":"):
                    pathname = os.path.normpath(pathname[:2]) + pathname[2:]
                    
                if (not check_for_directories) or os.path.isfile(pathname):
                    urls.append(pathname2url(pathname))
                    modpath = self.make_modpath_from_path(pathname)
                    self.add_files_to_modlist(modpath[:-1], [modpath[-1]],
                                              modlist)
                elif os.path.isdir(pathname):
                    W.walk_in_background(pathname,
                                         self.on_walk_callback,
                                         self.on_walk_completed)
            if len(modlist) > 0:
                self.file_collection_display.add(modlist)
            if len(urls) > 0 and self.workspace.file_list is not None:
                self.workspace.file_list.add_files_to_filelist(urls)
                
            
    def on_remove(self, mods):
        '''Called when the UI is asking to remove files from the list
        
        mods - two-tuple modpaths (see docs for FileCollectionDisplay)
        '''
        if self.workspace is not None:
            modpaths = sum([self.make_parts_list_from_mod(mod)
                            for mod in mods], [])
            urls = [self.modpath_to_url(modpath) for modpath in modpaths]
            hdf_file_list = self.workspace.get_file_list()
            hdf_file_list.remove_files_from_filelist(urls)
            self.file_collection_display.remove(mods)
            
    ext_dict = None
    def get_path_info(self, modpath):
        '''Get a descriptive name, the image type, the tooltip and menu for a path'''
        if self.ext_dict is None:
            self.ext_dict = {}
            #Order is important here, last in list will be the winner
            for (exts, node_type, menu) in (
                (SUPPORTED_IMAGE_EXTENSIONS, 
                 self.file_collection_display.NODE_MONOCHROME_IMAGE,
                 (self.MI_SHOW_IMAGE, self.MI_REMOVE)),
                (SUPPORTED_MOVIE_EXTENSIONS,
                 self.file_collection_display.NODE_MOVIE,
                 (self.MI_SHOW_IMAGE, self.MI_REMOVE)),
                ((".tif", ".ome.tiff", ".tiff"),
                 self.file_collection_display.NODE_MONOCHROME_IMAGE,
                 (self.MI_SHOW_IMAGE, self.MI_REMOVE)),
                ((".csv",), self.file_collection_display.NODE_CSV,
                 (self.MI_REMOVE,))):
                for ext in exts:
                    self.ext_dict[ext] = (node_type, menu)
                
        hdf_file_list = self.workspace.get_file_list()
        pathtype = hdf_file_list.get_type(self.modpath_to_url(modpath))
        if pathtype == hdf_file_list.TYPE_DIRECTORY:
            pathtype = self.file_collection_display.NODE_DIRECTORY
            menu = [self.MI_REMOVE, self.MI_REFRESH]
        else:
            # handle double-dot extensions: foo.bar.baz
            start = 0
            filename = modpath[-1]
            pathtype = self.file_collection_display.NODE_FILE
            menu = (self.MI_REMOVE,)
            while True:
                start = filename.find(".", start)
                if start == -1:
                    break
                x = self.ext_dict.get(filename[start:].lower(), None)
                if x is not None:
                    pathtype, menu = x
                    break
                start += 1
        return modpath[-1], pathtype, modpath[-1], menu
    
    def on_menu_command(self, path, command):
        '''Context menu callback
        
        This is called when the user picks a command from a context menu.
        
        path - a list of path parts to the picked item
        
        command - the command from the list supplied by get_path_info. None
                  means default = view image.
        '''
        if path[0] in ("http:", "https:", "ftp:"):
            url = path[0] + "//" + "/".join(path[1:])
            pathname = url
        else:
            pathname = os.path.join(*path)
            url = pathname2url(pathname)
        needs_raise_after = False
        if command is None:
            hdf_file_list = self.workspace.get_file_list()
            if hdf_file_list.get_type(url) == hdf_file_list.TYPE_FILE:
                command = self.MI_SHOW_IMAGE
                #%$@ assuming this is a double click, the new frame
                #    will be activated and then supplementary processing
                #    will set the focus back to the tree control, bringing
                #    the main window back to the front. Hence, we fight back
                #    by raising the window after the GUI has finished
                #    handling all events.
                #    Yes, I tried preventing further processing of events
                #    by the parent. Yes, this code is a disgusting hack.
                #    Yes, I hate the way GUI code turns your application
                #    into a giant pile of undecipherable spaghetti too.
                needs_raise_after = True
            else:
                return False
            
        if command == self.MI_SHOW_IMAGE:
            from cellprofiler.gui.cpfigure import CPFigureFrame
            from subimager.client import get_image
            filename = path[-1]
            try:
                image = get_image(url)
            except Exception, e:
                from cellprofiler.gui.errordialog import display_error_dialog
                display_error_dialog(None, e, None, 
                                     "Failed to load %s" % pathname)
            frame = CPFigureFrame(subplots = (1,1))
            if image.ndim == 2:
                frame.subplot_imshow_grayscale(0, 0, image, title = filename)
            else:
                frame.subplot_imshow_color(0, 0, image, title = filename)
            frame.Refresh()
            if needs_raise_after:
                #%$@ hack hack hack
                import wx
                wx.CallAfter(lambda: frame.Raise())
            return True
        elif command == self.MI_REFRESH:
            hdf_file_list = self.workspace.file_list
            modlist = []
            if hdf_file_list.get_type(url) == hdf_file_list.TYPE_DIRECTORY:
                urls = hdf_file_list.get_filelist(url)
            else:
                urls = [url]
            for url in urls:
                self.add_modpath_to_modlist(self.url_to_modpath(url), modlist)
            self.on_remove(modlist)
            if command == self.MI_REFRESH:
                W.walk_in_background(pathname,
                                     self.on_walk_callback,
                                     self.on_walk_completed)
            return True
    
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
        
    @classmethod
    def add_files_to_modlist(self, modpath, files, modlist = None):
        '''Add files to the tip of a modpath'''
        if modlist is None:
            modlist = []
        if len(modpath) == 1:
            modlist.append((modpath[0], files))
            return modlist
        idxs = [i for i, mod in enumerate(modlist)
                if isinstance(mod, tuple) and len(mod) == 2 and
                mod[0] == modpath[0]]
        if len(idxs) == 0:
            modlist.append((modpath[0], 
                            self.add_files_to_modlist(modpath[1:], files)))
            return modlist
        idx = idxs[0]
        self.add_files_to_modlist(modpath[1:], files, modlist[idx][1])
        return modlist
    
    @staticmethod
    def modpath_to_url(modpath):
        path = os.path.join(*modpath)
        return pathname2url(path)
    
    @staticmethod
    def url_to_modpath(url):
        if not url.lower().startswith("file:"):
            return None
        path = urllib.url2pathname(url[5:])
        parts = []
        while True:
            new_path, part = os.path.split(path)
            if len(new_path) == 0 or len(part) == 0:
                parts.insert(0, path)
                break
            parts.insert(0, part)
            path = new_path
        return parts
    
    def initialize_file_collection_display(self, workspace):
        d = {}
        hdf_file_list = workspace.file_list
        if hdf_file_list is None:
            return
        def fn(root, directories, files):
            if len(files) == 0:
                return
            if root.lower().startswith("file:") and len(root) > 5:
                path = urllib.url2pathname(root[5:])
                parts = []
                while True:
                    new_path, part = os.path.split(path)
                    if len(new_path) == 0 or len(part) == 0:
                        parts.append(path)
                        break
                    parts.append(part)
                    path = new_path
                dd = d
                for part in reversed(parts[1:]):
                    if not (dd.has_key(part) and dd[part] is not None):
                        dd[part] = {}
                    dd = dd[part]
                dd[parts[0]] = dict(
                    [(urllib.url2pathname(f), None) for f in files])
        hdf_file_list.walk(fn)
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
    
    @classmethod
    def make_modpath_from_path(cls, path):
        result = []
        while True:
            new_path, part = os.path.split(path)
            if len(new_path) == 0 or len(part) == 0:
                return [path] + result
            result.insert(0, part)
            path = new_path
            
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
            
    def filter_url(self, url):
        '''Return True if a URL passes the module's filter'''
        if not self.wants_filter:
            return True
        modpath = self.url_to_modpath(url)
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
            
    def change_causes_prepare_run(self, setting):
        '''Return True if a change to the settings requires a call to prepare_run
        
        Images should return True if any setting changes because that
        will affect the image plane descriptors passed onto later modules
        which will change the image set produced by the pipeline.
        '''
        return setting in self.settings()
    
    def is_input_module(self):
        return True
            
    def prepare_run(self, workspace):
        '''Create an IPD for every url that passes the filter'''
        if workspace.pipeline.in_batch_mode():
            return True
        urls = filter(self.filter_url, workspace.file_list.get_filelist())
        ipds = [cpp.ImagePlaneDetails(url, None, None, None) for url in urls]
        workspace.pipeline.add_image_plane_details(ipds, False)
        return True
        
    def run(self, workspace):
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
        lambda x: any([ExtensionPredicate.IS_TIF_PREDICATE(x), 
                       ExtensionPredicate.IS_JPEG_PREDICATE(x),
                       ExtensionPredicate.IS_PNG_PREDICATE(x)]), [],
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
    
    @staticmethod
    def is_stack(x):
        if (x.metadata.has_key(cpp.ImagePlaneDetails.MD_SIZE_T) and
              x.metadata[cpp.ImagePlaneDetails.MD_SIZE_T] > 1):
            return True
        if (x.metadata.has_key(cpp.ImagePlaneDetails.MD_SIZE_Z) and
            x.metadata[cpp.ImagePlaneDetails.MD_SIZE_Z] > 1):
            return True
        return False
        
    IS_STACK_PREDICATE = cps.Filter.FilterPredicate(
        "isstack", "Stack", lambda x: ImagePredicate.is_stack(x), [],
        doc = "The image is a Z-stack or movie")
    
    IS_STACK_FRAME_PREDICATE = cps.Filter.FilterPredicate(
        "isstackframe", "Stack frame", lambda x: x.index is not None, [],
        doc = "The image is a frame of a movie or a plane of a Z-stack")
    
    def __init__(self):
        subpredicates = ( self.IS_COLOR_PREDICATE, 
                          self.IS_MONOCHROME_PREDICATE,
                          self.IS_STACK_PREDICATE,
                          self.IS_STACK_FRAME_PREDICATE)
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

    class FakeModule(cpm.CPModule):
        '''A fake module for setting validation'''
        def get_image_plane_details(self, modpath):
            url = Images.modpath_to_url(modpath)
            return cpp.ImagePlaneDetails(url, None, None, None)
    def test_valid(self, pipeline, *args):
        self((cps.FileCollectionDisplay.NODE_FILE, 
              ["/imaging", "test.tif"], self.FakeModule()), *args)

