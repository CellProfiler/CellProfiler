# coding=utf-8

import cellprofiler
import cellprofiler.gui.help
import cellprofiler.icons
import cellprofiler.modules
import cellprofiler.preferences
import glob
import os
import re
import shutil
import sys
from shutil import copy

import cellprofiler.icons
import cellprofiler.preferences as cpprefs
from cellprofiler.gui.help import MAIN_HELP
from cellprofiler.modules import get_module_names, instantiate_module
import os.path

LOCATION_COVERPAGE = os.path.join('images', 'CPCoverPage.png')
LOCATION_WHITEHEADLOGO = os.path.join('images', 'WhiteheadInstituteLogo.png')
LOCATION_CSAILLOGO = os.path.join('images', 'CSAIL_Logo.png')
LOCATION_IMAGINGPLATFORMBANNER = os.path.join('images', 'BroadPlusImagingPlusBanner.png')
VERSION = cellprofiler.__version__
VERSION_NUMBER = int(re.sub(r"\.|rc\d{1}", "", cellprofiler.__version__))


def generate_html(webpage_path=None):
    if webpage_path is None:
        webpage_path = os.path.join('.', 'CellProfiler_Manual_' + str(VERSION_NUMBER))

    if not (os.path.exists(webpage_path) and os.path.isdir(webpage_path)):
        os.mkdir(webpage_path)

    # Copy the png images to a new 'images' directory under the html folder
    webpage_images_path = os.path.join(webpage_path, 'images')
    if not (os.path.exists(webpage_images_path) and os.path.isdir(webpage_images_path)):
        os.mkdir(webpage_images_path)

    # Write the individual topic files
    module_help_text = output_module_html(webpage_path)
    nonmodule_help_text = output_gui_html(webpage_path)

    index_fd = open(os.path.join(webpage_path, 'index.html'), 'w')

    icons_path = cellprofiler.icons.__path__[0]
    all_pngs = glob.glob(os.path.join(icons_path, "*.png"))
    for f in all_pngs:
        shutil.copy(f, webpage_images_path)

    intro_text = """
<html style="font-family:arial">
<head>
<title>CellProfiler: Table of contents</title>
</head>
<body>
<div style="page-break-after:always">
<table width="100%%">
<tr><td align="center">
<img src="%(LOCATION_COVERPAGE)s" align="middle" style="border-style: none">
</tr></td>
</table>
</div>
<div style="page-break-after:always">
<table width="100%%" cellpadding="10">
<tr><td align="middle"><b>CellProfiler</b> cell image analysis software</td></tr>
<tr><td align="middle"><b>Created by</b><br>Anne E. Carpenter and Thouis R. Jones</td></tr>
<tr><td align="middle"><b>In the laboratories of</b><br>David M. Sabatini and Polina Golland at</td></tr>
<tr><td align="middle"><img src="%(LOCATION_WHITEHEADLOGO)s" style="border-style: none">
<img src="%(LOCATION_CSAILLOGO)s" style="border-style: none"></td></tr>
<tr><td align="middle">And now based at</td></tr>
<tr><td align="middle"><img src="%(LOCATION_IMAGINGPLATFORMBANNER)s" style="border-style: none"></td></tr>
<tr><td align="middle">
<b>CellProfiler is free and open-source!</b>

<p>If you find it useful, please credit CellProfiler in publications
<ol>
<li>Cite the <a href="www.cellprofiler.org">website</a>.</li>
<li>Cite the <a href="http://cellprofiler.org/citations.html">publication</a>.</li>
<li>Post the reference for your publication on the CellProfiler <a href="http://forum.cellprofiler.org/">forum</a> so that we are aware of it.</li>
</ol></p>

<p>These steps will help us to maintain funding for the project and continue to
improve and support it.</p>
</td></tr>
</table>
</div>

<b>This manual accompanies version %(VERSION)s of CellProfiler. The most
recent manual is available <a href="http://d1zymp9ayga15t.cloudfront.net/CPmanual/index.html">here</a>.</b>

<h1><a name="table_of_contents">Table of contents</a></h1>""" % globals()

    index_fd.write(intro_text)
    index_fd.write(nonmodule_help_text)
    index_fd.write(module_help_text)
    index_fd.write("""</body></html>\n""")

    index_fd.close()

    print("Wrote CellProfiler Manual to", webpage_path)


def output_gui_html(webpage_path):
    """Output an HTML page for each non-module help item"""
    icons_relpath = os.path.relpath(cellprofiler.icons.__path__[0])

    help_text = """
<h2>Using CellProfiler</a></h2>"""

    def write_menu(prefix, h, help_text):
        help_text += "<ul>\n"
        for key, value in h:
            help_text += "<li>"
            if hasattr(value, "__iter__") and not isinstance(value, str):
                help_text += "<b>%s</b>" % key
                help_text = write_menu(prefix + "_" + key, value, help_text)
            else:
                # Replace special characters with blanks
                cleaned_up_key = re.sub("[/\\\?%\*:\|\"<>\.\+]", "", key)
                # Replace spaces with underscores
                cleaned_up_key = re.sub(" ", "_", cleaned_up_key)
                file_name = "%s_%s.html" % (prefix, cleaned_up_key)

                fd = open(os.path.join(webpage_path, file_name), "w")
                fd.write("<html style=""font-family:arial""><head><title>%s</title></head>\n" % key)
                fd.write("<body><h1>%s</h1>\n<div>\n" % key)

                # Replace the relative paths to the icons with the relative path to the image dir
                value = value.replace(icons_relpath, 'images')
                # Replace refs to icons in memory with the relative path to the image dir
                #  Slashes need to be escaped: http://stackoverflow.com/questions/4427174/python-re-bogus-escape-error
                value = re.sub("memory:", os.path.join("images", "").encode('string-escape'), value)

                fd.write(value)
                fd.write("</div></body>\n")
                fd.close()
                help_text += "<a href='%s'>%s</a>\n" % (file_name, key)
            help_text += "</li>\n"
        help_text += "</ul>\n"
        return help_text

    help_text = write_menu("Help", cellprofiler.gui.help.MAIN_HELP, help_text)
    help_text += "\n"

    return help_text


def output_module_html(webpage_path):
    """Output an HTML page for each module"""

    icons_relpath = os.path.relpath(cellprofiler.icons.__path__[0])
    all_png_icons = glob.glob(os.path.join(icons_relpath, "*.png"))
    icon_names = [os.path.basename(f)[:-4] for f in all_png_icons]

    help_text = """
<h2>Help for CellProfiler Modules</a></h2>
<ul>\n"""
    d = {}
    module_path = webpage_path
    if not (os.path.exists(module_path) and os.path.isdir(module_path)):
        try:
            os.mkdir(module_path)
        except IOError:
            raise ValueError("Could not create directory %s" % module_path)

    for module_name in sorted(cellprofiler.modules.get_module_names()):
        module = cellprofiler.modules.instantiate_module(module_name)
        location = os.path.split(
                module.create_settings.__func__.__code__.co_filename)[0]
        if location == cellprofiler.preferences.get_plugin_directory():
            continue
        if isinstance(module.category, str):
            module.category = [module.category]
        for category in module.category:
            if category not in d:
                d[category] = {}
            d[category][module_name] = module
        result = module.get_help()
        if result is None:
            continue
        result = result.replace('<body><h1>', '<body><h1>Module: ')

        # Replace refs to icons in memory with the relative path to the image dir (see above)
        result = re.sub("memory:", os.path.join("images", "").encode('string-escape'), result)

        # Check if a corresponding image exists for the module
        if module_name in icon_names:
            # Strip out end html tags so I can add more stuff
            result = result.replace('</body>', '').replace('</html>', '')

            # Include images specific to the module, relative to html files ('images' dir)
            LOCATION_MODULE_IMAGES = os.path.join('images', '%s.png' % module_name)
            result += '\n\n<div><p><img src="%s", width="50%%"></p></div>\n' % LOCATION_MODULE_IMAGES

            # Now end the help text
            result += '</body></html>'
        fd = open(os.path.join(module_path, "%s.html" % module_name), "w")
        fd.write(result)
        fd.close()
    for category in sorted(d.keys()):
        sub_d = d[category]
        help_text += "<li><b>%s</b><br><ul>\n" % category
        for module_name in sorted(sub_d.keys()):
            help_text += "<li><a href='%s.html'>%s</a></li>\n" % (module_name, module_name)
        help_text += "</ul></li>\n"
    help_text += "</ul>\n"
    return help_text


def search_module_help(text):
    """Search the help for a string

    text - find text in the module help using case-insensitive matching

    returns an html document of all the module help pages that matched or
            None if no match found.
    """
    matching_help = []
    for item in cellprofiler.gui.help.MAIN_HELP:
        matching_help += __search_menu_helper(
                item, lambda x: __search_fn(x, text))
    count = sum([len(x[2]) for x in matching_help])

    for module_name in cellprofiler.modules.get_module_names():
        module = cellprofiler.modules.instantiate_module(module_name)
        location = os.path.split(
                module.create_settings.__func__.__code__.co_filename)[0]
        if location == cellprofiler.preferences.get_plugin_directory():
            continue
        help_text = module.get_help()
        matches = __search_fn(help_text, text)
        if len(matches) > 0:
            matching_help.append((module_name, help_text, matches))
            count += len(matches)
    if len(matching_help) == 0:
        return None
    top = """<html style="font-family:arial">
    <head><title>%s found</title></head>
    <body><h1>Matches found</h1><br><ul>
    """ % ("1 match" if len(matching_help) == 1 else "%d matches" % len(matching_help))
    body = "<br>"
    match_num = 1
    prev_link = (
        '<a href="#match%d" title="Previous match">'
        '<img src="memory:previous.png" alt="previous match"></a>')
    anchor = '<a name="match%d"><u>%s</u></a>'
    next_link = ('<a href="#match%d" title="Next match">'
                 '<img src="memory:next.png" alt="next match"></a>')
    for title, help_text, pairs in matching_help:
        top += """<li><a href="#match%d">%s</a></li>\n""" % (
            match_num, title)
        if help_text.find("<h1>") == -1:
            body += "<h1>%s</h1>" % title
        start_match = re.search(r"<\s*body[^>]*?>", help_text, re.IGNORECASE)
        if start_match is None:
            start = 0
        else:
            start = start_match.end()
        end_match = re.search(r"<\\\s*body", help_text, re.IGNORECASE)
        if end_match is None:
            end = len(help_text)
        else:
            end = end_match.start()

        for begin_pos, end_pos in pairs:
            body += help_text[start:begin_pos]
            if match_num > 1:
                body += prev_link % (match_num - 1)
            body += anchor % (match_num, help_text[begin_pos:end_pos])
            if match_num != count:
                body += next_link % (match_num + 1)
            start = end_pos
            match_num += 1
        body += help_text[start:end] + "<br>"
    result = "%s</ul><br>\n%s</body></html>" % (top, body)
    return result


def __search_fn(html, text):
    """Find begin-end coordinates of case-insensitive matches in html

    html - an HTML document

    text - a search string

    Find the begin and end indices of case insensitive matches of "text"
    within the text-data of the HTML, searching only in its body and excluding
    text in the HTML tags.
    """
    start_match = re.search(r"<\s*body[^>]*?>", html, re.IGNORECASE)
    if start_match is None:
        start = 0
    else:
        start = start_match.end()
    end_match = re.search(r"<\\\s*body", html, re.IGNORECASE)
    if end_match is None:
        end = len(html)
    else:
        end = end_match.start()
    escaped_text = re.escape(text)
    if " " in escaped_text:
        #
        # Many problems here:
        # <b>Groups</b> module
        # Some\ntext
        #
        # For now, just solve the multiple space problems
        #
        escaped_text = escaped_text.replace("\\ ", "\\s+")
    pattern = "(<[^>]*?>|%s)" % escaped_text
    return [(x.start() + start, x.end() + start)
            for x in re.finditer(pattern, html[start:end], re.IGNORECASE)
            if x.group(1)[0] != '<']


def __search_menu_helper(menu, search_fn):
    """Search a help menu for text

    menu - a menu in the style of MAIN_HELP. A leaf is a two-tuple composed
           of a title string and its HTML help. Non-leaf branches are two-tuples
           of titles and tuples of leaves and branches.

    search_fn - given a string, returns a list of begin-end tuples of search
                matches within that string.

    returns a list of three-tuples. The first item is the title. The second is
    the html help. The third is a list of begin-end tuples of matches found.
    """
    if len(menu) == 2 and all([isinstance(x, str) for x in menu]):
        matches = search_fn(menu[1])
        if len(matches) > 0:
            return [(menu[0], menu[1], matches)]
        return []
    return sum([__search_menu_helper(x, search_fn) for x in menu[1]], [])
