"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
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
import cellprofiler.preferences as cpprefs
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw
import cellprofiler.utilities.walk_in_background as W
import cellprofiler.utilities.jutil as J
import os
import sys
import urllib
import uuid

from .loadimages import pathname2url, SUPPORTED_IMAGE_EXTENSIONS
from .loadimages import SUPPORTED_MOVIE_EXTENSIONS
from cellprofiler.utilities.hdf5_dict import HDF5FileList

class Images(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Images"
    category = "File Processing"
    
    MI_SHOW_IMAGE = "Show image"
    MI_REMOVE = cps.FileCollectionDisplay.DeleteMenuItem("Remove from list")
    MI_REFRESH = "Refresh"
    
    def create_settings(self):
        self.workspace = None
        self.module_explanation = cps.HTMLText("",content="""
            To begin creating your workspace, use the %s module to compile 
            a list of files and/or folders that you want to analyze. You can also specify a set of rules 
            to include only the desired files in your selected folders."""%self.module_name,
            size=(0, 2.3))      
        
        self.path_list_display = cps.PathListDisplay()
        predicates = [FilePredicate(),
                      DirectoryPredicate(),
                      ExtensionPredicate()]
        self.wants_filter = cps.Binary(
            "Filter based on rules", False,
            doc = "Check this setting to display and use the rules filter")
            
        self.filter = cps.Filter("Filter", predicates, 
                                 'or (file does contain "")')
        self.update_button = cps.PathListRefreshButton(
            "Update filter", "Update",
            doc = """<i>(Only displayed if filtering based on rules)</i>
            Redisplay the file list, removing or graying out the files
            that do not pass the current filter.
            """)
    
    @staticmethod
    def modpath_to_url(modpath):
        if modpath[0] in ("http", "https", "ftp"):
            if len(modpath) == 1:
                return modpath[0] + ":"
            elif len(modpath) == 2:
                return modpath[0] + ":" + modpath[1] 
            else:
                return modpath[0] + ":" + modpath[1] + "/" + "/".join(
                    [urllib.quote(part) for part in modpath[2:]])
        path = os.path.join(*modpath)
        return pathname2url(path)
    
    @staticmethod
    def url_to_modpath(url):
        if not url.lower().startswith("file:"):
            schema, rest = HDF5FileList.split_url(url)
            return [schema] + rest[0:1] + [urllib.unquote(part) for part in rest[1:]]
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
    
    @classmethod
    def make_modpath_from_path(cls, path):
        result = []
        while True:
            new_path, part = os.path.split(path)
            if len(new_path) == 0 or len(part) == 0:
                return [path] + result
            result.insert(0, part)
            path = new_path
            
    def filter_url(self, url):
        '''Return True if a URL passes the module's filter'''
        if not self.wants_filter:
            return True
        
        return evaluate_url(self.filter, url)
    
    def settings(self):
        return [self.path_list_display, self.wants_filter, self.filter]
    
    def visible_settings(self):
        result = [self.module_explanation,self.path_list_display, self.wants_filter]
        if self.wants_filter:
            result += [self.filter, self.update_button]
        return result
            
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
        workspace.pipeline.load_image_plane_details(workspace)
        if self.wants_filter:
            env = J.get_env()
            jexpression = env.new_string_utf(self.filter.value_text)
            filter_fn = J.make_static_call(
                "org/cellprofiler/imageset/filter/Filter",
                "filter", "(Ljava/lang/String;Ljava/lang/String;)Z")
            ipds = filter(
                lambda ipd:filter_fn(jexpression, env.new_string_utf(ipd.url)), 
                workspace.pipeline.image_plane_details)
        else:
            ipds = workspace.pipeline.image_plane_details
        workspace.pipeline.set_filtered_image_plane_details(ipds, self)
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


filter_class = None
filter_method_id = None
def evaluate_url(filtr, url):
    '''Evaluate a URL using a setting's filter
    
    '''
    global filter_class, filter_method_id
    env = J.get_env()
    if filter_class is None:
        filter_class = env.find_class("org/cellprofiler/imageset/filter/Filter")
        if filter_class is None:
            jexception = get_env.exception_occurred()
            raise J.JavaException(jexception)
        filter_method_id = env.get_static_method_id(
            filter_class,
            "filter", 
            "(Ljava/lang/String;Ljava/lang/String;)Z")
        if filter_method_id is None:
            raise JavaError("Could not find static method, org.cellprofiler.imageset.filter.Filter.filter(String, String)")
    try:
        expression = filtr.value_text
        if isinstance(expression, unicode):
            expression = expression.encode("utf-8")
        if isinstance(url, unicode):
            url = url.encode("utf-8")
        return env.call_static_method(
            filter_class, filter_method_id, 
            env.new_string_utf(expression), env.new_string_utf(url))
    except J.JavaException as e:
        if J.is_instance_of(
            e.throwable, "org/cellprofiler/imageset/filter/Filter$BadFilterExpressionException"):
            raise
        return False
    