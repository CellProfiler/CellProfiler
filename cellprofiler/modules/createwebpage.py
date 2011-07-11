'''<b>Create Web Page</b> creates the html file for a webpage to display images 
(or their thumbnails, if desired)
<hr>

This module creates an html file that displays the specified
images, and optionally a link to a compressed ZIP file of all of the images shown.
'''

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2011
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__ = "$Revision$"

import os
import sys
import zipfile

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs
from cellprofiler.modules.loadimages import C_FILE_NAME, C_PATH_NAME
from cellprofiler.gui.help import USING_METADATA_TAGS_REF, USING_METADATA_HELP_REF

DIR_ABOVE = "One level over the images"
DIR_SAME = "Same as the images"

OPEN_ONCE = "Once only"
OPEN_EACH = "For each image"
OPEN_NO = "No"

'''Use this dictionary to maintain backwards compatibility.
If you change one of the wordings for the choices, enter the new wording
for the choice as a key in the dictionary and the symbol for that key
as the value.'''
TRANSLATION_DICTIONARY = {
    "One level over the images": DIR_ABOVE,
    "Same as the images": DIR_SAME,
    "Once only": OPEN_ONCE,
    "For each image": OPEN_EACH,
    "No": OPEN_NO
    }

#
# os.path.relpath is only available from Python 2.6 +
# The following is derived from posixpath.py from the Python distribution
#
if hasattr(os.path, "relpath"):
    relpath = os.path.relpath
else:
    def relpath(path, start = os.curdir):
        start = os.path.abspath(start)
        path = os.path.abspath(path)
        if sys.platform.startswith("win"):
            start = start.lower()
            path = path.lower()
            #
            # Check to see if UNC paths match ("\\foo\bar" style)
            #
            start_unc = os.path.splitunc(start)[0]
            path_unc = os.path.splitunc(path)[0]
            if start_unc != path_unc:
                raise ValueError("Can't get relative path between %s and %s"
                                 (start, path))
            if len(start_unc):
                # Drive letters.
                start_drive = os.path.splitdrive(start)[0]
                path_drive = os.path.splitdrive(path)[0]
                if start_drive != path_drive:
                    raise ValueError("Can't get relative path between %s and %s"
                                     (start, path))
        start_list = start.split(os.path.sep)
        path_list = path.split(os.path.sep)
        i = len(os.path.commonprefix((start_list, path_list)))
        rel_list = [os.pardir] * (len(start_list) - i) + path_list[i:]
        if len(rel_list) == 0:
            return os.curdir
        return '/'.join(rel_list)

class CreateWebPage(cpm.CPModule):
    
    module_name = "CreateWebPage"
    category = "Other"
    variable_revision_number = 1
    
    def create_settings(self):
        self.orig_image_name = cps.ImageNameSubscriber(
            "Select the input images", "None",
            doc="""Select the images to display on the web page.""")
        
        self.wants_thumbnails = cps.Binary(
            "Use thumbnail images?", False,
            doc="""Check this option to display thumbnail images (small versions of the 
            images) on the web page that link to the full images. Leave it 
            unchecked to display the full image directly on the web page.
            <p>If you are going to use thumbnails, you will need to load
            them using <b>LoadImages</b> or <b>LoadData</b>; you can run a separate 
            pipeline prior to this one to create thumbnails from your originals  
            using the <b>Resize</b> and <b>SaveImages</b> modules. For some high-content
            screening systems, thumbnail files are automatically created and have
            the text "thumb" in the name.</p>""")
        
        self.thumbnail_image_name = cps.ImageNameSubscriber(
            "Select the thumbnail images", "None",
            doc="""
            <i>(Used only if using thumbnails)</i><br>
            Select the name of the images to use for thumbnails.""")
        
        self.web_page_file_name = cps.Text(
            "Webpage file name", "images1",
            metadata= True,
            doc="""Enter the desired file name for the web page. <b>CreateWebPage</b>
            will add the .html extension if no extension is specified.
            If you have metadata associated with your images, you can name the 
            file using metadata tags. %(USING_METADATA_TAGS_REF)s<br>
            For instance, if you have metadata tags named "Plate" and 
            "Well", you can create separate per-plate, per-well web pages based on
            your metadata by inserting the tags "Plate_Well" to specify the 
            name. %(USING_METADATA_HELP_REF)s."""%globals())
        
        self.directory_choice = cps.Choice(
            "Select the folder for the .html file",
            [ DIR_SAME, DIR_ABOVE],
            doc="""This setting determines how <b>CreateWebPage</b> selects the 
            folder for the .html file(s) it creates. 
            <ul>
            <li><i>%(DIR_SAME)s</i>: Place the .html file(s) in the same folder as 
            the files.</li>
            <li><i>%(DIR_ABOVE)s</i>: Place the .html file(s) in the
            image files' parent folder.</li>
            </ul>""" % globals())
        
        self.title = cps.Text(
            "Webpage title", "Image", metadata = True,
            doc = """This is the title that appears at the top of the browser
            window. If you have metadata associated with your images, you can name the 
            file using metadata tags. %(USING_METADATA_TAGS_REF)sFor instance, if you 
            have a metadata tag named "Plate", you can type "Plate: " and then insert 
            the metadata tag "Plate" to display the plate metadata item. %(USING_METADATA_HELP_REF)s."""
            %globals())
        
        self.background_color = cps.Color(
            "Webpage background color", "White",
            doc = """This setting controls the background color for the web page.""")
        
        self.columns = cps.Integer(
            "Number of columns", 1, minval = 1,
            doc = """This setting determines how many images are displayed
            in each row.""")
        
        self.table_border_width = cps.Integer(
            "Table border width", 1, minval = 0,
            doc = """The table border width determines the width of the border
            around the entire grid of displayed images (i.e., the "table" of images) and is measured in pixels. This value can be 
            set to zero, in which case you will not see the table border.""")
        
        self.table_border_color = cps.Color(
            "Table border color", "White")

        self.image_spacing = cps.Integer(
            "Image spacing", 1, minval = 0,
            doc = """The spacing between images ("table cells"), in pixels.""")
        
        self.image_border_width = cps.Integer(
            "Image border width", 1, minval = 0,
            doc = """The image border width determines the width of
            the border around each image and is measured in pixels.
            This value can be set to zero, in which case you will not see the 
            image border.""")
        
        self.create_new_window = cps.Choice(
            "Open new window when viewing full image?",
            [OPEN_ONCE, OPEN_EACH, OPEN_NO],
            doc = """This controls the behavior of the thumbnail links. If
            you select, <i>%(OPEN_ONCE)s</i>, your browser will open a new window
            when you click on the first thumbnail and will display subsequent
            images in the newly opened window. If you select <i>
            %(OPEN_EACH)s</i>, the browser will open a new window each time
            you click on a link. If you select, <i>%(OPEN_NO)s</i>, the
            browser will reuse the current window to display the image"""% globals())
        
        self.wants_zip_file = cps.Binary(
            "Make a ZIP file containing the full-size images?", False,doc="""
            ZIP files are a common archive and data compression file format, making it convenient to download all of the images represented on the web page with a single click.
            Check this box to create a ZIP file that contains all your images, 
            compressed to reduce file size.""")
        
        self.zipfile_name = cps.Text(
            "Enter the ZIP file name", "Images.zip",
            metadata = True,
            doc="""
            <i>(Used only if creating a ZIP file)</i><br>
            Specify the name for the ZIP file.""")
        
    def settings(self):
        '''The settings as saved in the pipeline'''
        return [self.orig_image_name, self.wants_thumbnails, 
                self.thumbnail_image_name, self.web_page_file_name,
                self.directory_choice, self.title, self.background_color,
                self.columns, self.table_border_width, self.table_border_color,
                self.image_spacing,
                self.image_border_width, self.create_new_window, 
                self.wants_zip_file, self.zipfile_name]
    
    def visible_settings(self):
        '''the settings as displayed in the gui'''
        result = [self.orig_image_name, self.wants_thumbnails]
        if self.wants_thumbnails:
            result += [self.thumbnail_image_name]
        result += [self.web_page_file_name, self.directory_choice,
                   self.title,
                   self.background_color, self.columns,
                   self.table_border_width, self.table_border_color,
                   self.image_spacing,
                   self.image_border_width, self.create_new_window,
                   self.wants_zip_file]
        if self.wants_zip_file:
            result += [self.zipfile_name]
        return result
    
    def validate_module(self, pipeline):
        '''Make sure metadata tags exist'''
        for cntrl in (self.web_page_file_name, self.title): 
            undefined_tags = pipeline.get_undefined_metadata_tags(cntrl.value)
            if len(undefined_tags) > 0:
                raise cps.ValidationError("%s is not a defined metadata tag. Check the metadata specifications in your load modules" %
                                 undefined_tags[0], 
                                 cntrl)
                
    def is_interactive(self):
        return False

    def get_filenames(self, image_set_list):
        '''Return the set of file names'''
        return self.get_dictionary(image_set_list)["Filenames"]
        
    def prepare_run(self, workspace):
        '''Initialize the dictionary of file names here'''
        d = self.get_dictionary(workspace.image_set_list)
        d["Filenames"] = {}
        return True
        
    def run(self, workspace):
        '''Process the current image set - write the image links to the web page
        
        workspace - workspace for current image set
        '''
        
        m = workspace.measurements
        statistics = []
        if workspace.frame is not None:
            workspace.display_data.statistics = statistics
            
        image_name = self.orig_image_name.value
        image_path_name, image_file_name = self.get_image_location(
            workspace, image_name)
        abs_image_path_name = os.path.abspath(os.path.join(image_path_name,
                                                           image_file_name))
        if self.directory_choice == DIR_ABOVE:
            path_name, image_path_name = os.path.split(image_path_name)
            image_path_name = '/'.join((image_path_name, image_file_name))
        else:
            path_name = image_path_name
            image_path_name = image_file_name
        
        if self.wants_thumbnails:
            thumbnail_image_name = self.thumbnail_image_name.value
            thumbnail_path_name, thumbnail_file_name = self.get_image_location(
                workspace, thumbnail_image_name)
            #
            # Make the thumbnail path name relative to the location for
            # the .html file
            #
            thumbnail_path_name = relpath(thumbnail_path_name, path_name)
            if os.path.sep != '/':
                thumbnail_path_name = thumbnail_path_name.replace(
                    os.path.sep, '/')
            thumbnail_path_name = '/'.join((thumbnail_path_name, 
                                           thumbnail_file_name))
            
        file_name = self.web_page_file_name.value
        file_name = m.apply_metadata(file_name)
        if file_name.find('.') == -1:
            file_name += ".html"
        file_path = os.path.join(path_name, file_name)
        statistics += [["HTML file", file_path],
                       ["Image file", image_file_name]]
        if self.wants_thumbnails:
            statistics += [["Thumbnail file", thumbnail_path_name]]
                       
        d = self.get_filenames(workspace.image_set_list)
        if self.wants_zip_file:
            zip_file_name = self.zipfile_name.value
            zip_file_name = m.apply_metadata(zip_file_name)
            if zip_file_name.find('.') == -1:
                zip_file_name += ".zip"
            zip_file_path = os.path.join(path_name, zip_file_name)
            mode = "a" if os.path.exists(zip_file_path) else "w"
            z = zipfile.ZipFile(zip_file_path, mode)
            z.write(abs_image_path_name)
            z.close()
            statistics += [["Zip file", zip_file_path]]
        if not file_path in d.keys():
            #
            # Here, we make a new file, including HTML header
            #
            d[file_path] = { "column": 0 }
            fd = open(file_path, "w")
            title = m.apply_metadata(self.title.value)
            bgcolor = self.background_color.value.replace(' ', '')
            table_border_width = self.table_border_width.value
            table_border_color = self.table_border_color.value.replace(' ','')
            cell_spacing = self.image_spacing.value
            fd.write("""<html>
<head><title>%(title)s</title></head>
<body bgcolor='%(bgcolor)s'>\n""" % locals())
            if self.wants_thumbnails:
                fd.write("""<div>Click an image to see a higher-resolution version.</div>\n""")
            if self.wants_zip_file:
                fd.write("""<center><a href='%s'>Download all high-resolution 
    images as a zipped file.</a></center><p>\n""" % zip_file_name)
            fd.write("""<center>
<table border='%(table_border_width)d' 
       bordercolor='%(table_border_color)s'
       cellpadding='0'
       cellspacing='%(cell_spacing)s'>""" % locals())
        else:
            fd = open(file_path, "a")
        column = d[file_path]["column"]
        if column == 0:
            fd.write("<tr>\n")
        image_border_width = self.image_border_width.value
        fd.write("<td>\n")
        if self.wants_thumbnails:
            #
            # Get the target attribute for the link - tells which
            # window will open when clicked. "_blank" = new window each time
            #
            if self.create_new_window == OPEN_ONCE:
                target = " target='_CPNewWindow'"
            elif self.create_new_window == OPEN_EACH:
                target = " target='_blank'"
            else:
                target = ""
            #
            # Write a link tag going to the real image with an image
            # tag pointing at the thumbnail
            #
            fd.write("""<a href='%(image_path_name)s' %(target)s>
    <img src='%(thumbnail_path_name)s' border='%(image_border_width)d' /></a>\n""" %
                     locals())
        else:
            #
            # Write an image tag pointing at the image
            #
            fd.write("""<img src='%(image_path_name)s' 
            border='%(image_border_width)d' />\n""" % locals())
        fd.write("</td>\n")
        column += 1
        if column == self.columns:
            fd.write("</tr>\n")
            column = 0
        d[file_path]["column"] = column
        fd.close()

    def display(self, workspace):
        '''Put up a simple display of web file statistics'''
        figure = workspace.create_or_find_figure(title="CreateWebPage, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(1,1))
        figure.subplot_table(0, 0, workspace.display_data.statistics)
        
    def post_run(self, workspace):
        '''Write the last few tags for each file at the end of the run'''
        d = self.get_filenames(workspace.image_set_list)
        for key in d:
            fd = open(key, "a")
            if d[key]["column"] != 0:
                fd.write("</tr>\n")
            fd.write("</table></center>\n</body>\n</html>\n")
            fd.close()
        
    def get_image_location(self, workspace, image_name):
        '''Get the path and file name for an image
        
        workspace - workspace for current image set
        image_name - image whose path should be fetched
        '''
        file_name_feature = '_'.join((C_FILE_NAME, image_name))
        path_name_feature = '_'.join((C_PATH_NAME, image_name))
        m = workspace.measurements
        image_file_name = m.get_current_image_measurement(file_name_feature)
        image_path_name = m.get_current_image_measurement(path_name_feature)
        return image_path_name, image_file_name
        
    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if variable_revision_number == 1 and from_matlab:
            orig_image, thumb_image, create_ba, file_name, directory_option, \
            page_title, bg_color, thumb_cols, table_border_width, \
            table_border_color, thumb_spacing, thumb_border_width, \
            create_new_window, zip_file_name = setting_values
            wants_thumbnails = thumb_image != cps.DO_NOT_USE
            wants_zip_file = zip_file_name != cps.DO_NOT_USE
            
            setting_values = [ 
                orig_image, wants_thumbnails, thumb_image, file_name,
                directory_option, page_title, bg_color, thumb_cols,
                table_border_width, table_border_color, thumb_spacing,
                thumb_border_width, create_new_window, wants_zip_file,
                zip_file_name]
            from_matlab = False
            variable_revision_number = 1
            
        setting_values = list(setting_values)
        for index in (4, 12):
            setting_values[index] = TRANSLATION_DICTIONARY[setting_values[index]]
            
        return setting_values, variable_revision_number, from_matlab
