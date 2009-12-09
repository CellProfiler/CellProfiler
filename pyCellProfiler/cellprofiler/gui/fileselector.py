import wx
import wx.html
import wx.lib.agw.customtreectrl as CT
import os
import os.path
import fnmatch
import re
import sys

default_input = '/tmp'
default_output = '/Users/thouis'
try:
    default_input = sys.argv[1]
    default_output = sys.argv[2]
except:
    pass

base_dir_choices = ['Default Image directory', 'Default Output directory', 'Other Directory...']
FS_DEFAULT_IMAGE, FS_DEFAULT_OUTPUT, FS_OTHER_DIR = base_dir_choices

descend_dir_choices = ['No', 'Yes', 'Choose...']
FS_DESCEND_NO, FS_DESCEND_YES, FS_DESCEND_CHOOSE = descend_dir_choices

match_modes = ['Common Substring', 'Shell pattern', 'Regular expression', 'Numeric Position']
FS_SUBSTRING, FS_SHELL, FS_RE, FS_POSITION = match_modes
pattern_label_strings = ["Substring these images have in common: ",
                         "Shell pattern (include leading *, if needed): ",
                         "Regular expression: ",
                         "Position "]

match_elements = ['Filename only', 'Subdirectory and filename', 'Full path']
FS_FILENAME_ONLY, FS_SUBDIR_AND_FNAME, FS_FULL_PATH = match_elements


default_image_names = ['DNA', 'Actin', 'Protein']

def relpath(sub, parent):
    tails = []
    assert sub.startswith(parent)
    while sub.startswith(parent):
        sub, tail = os.path.split(sub)
        tails.append(tail)
    tails.reverse()
    # drop the first element, as that one was part of parent
    if len(tails) > 1:
        return os.path.join(*tails[1:])
    return ''
        
def default_image_name(idx):
    try:
        return default_image_names[idx]
    except:
        return 'Image%d'%(idx + 1)

myEVT_CUSTOM_EVENT = wx.NewEventType()
EVT_CUSTOM_EVENT = wx.PyEventBinder(myEVT_CUSTOM_EVENT, 1)

class MyEvent(wx.PyEvent):
    def __init__(self):
        wx.PyEvent.__init__(self)
        self.SetEventType(myEVT_CUSTOM_EVENT)


class CPFileSelector(wx.Frame):
    def __init__(self, *args, **kwds):

        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)

        # controls

        def labeled_thing(label, thing):
            text = wx.StaticText(self, -1, label)
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(text, 0, wx.ALIGN_CENTER)
            sizer.AddSpacer(5)
            sizer.Add(thing, 0, wx.ALIGN_CENTER)
            return thing, sizer
        
        def boxed_thing(box, thing, flag=0):
            boxsizer = wx.StaticBoxSizer(box, wx.VERTICAL)
            boxsizer.Add(thing, 1, flag=flag)
            return thing, boxsizer

        # Base directory
        base_dir, base_dir_sizer = labeled_thing("Where are your images located?", wx.Choice(self, -1, choices=base_dir_choices))
        self.base_dir = base_dir
        # otherdir entry & browse
        otherdir_label = wx.StaticText(self, -1, "Other directory:")
        self.otherdir = otherdir = wx.TextCtrl(self, -1, "")
        otherdir_browse = wx.Button(self, -1, "Browse...")

        # descend?
        descend_dirs, descend_sizer = labeled_thing("Descend into subdirectories?", wx.Choice(self, -1, choices=descend_dir_choices))
        self.descend_dirs = descend_dirs
        # descend force update
        self.descend_update_filelist = descend_update_filelist = wx.Button(self, -1, "Update file list...")
        # descent tree chooser...
        box = wx.StaticBox(self, -1, "Choose directories to search for files...")
        self.dirtree, self.dirtree_boxsizer = dirtree, dirtree_boxsizer = boxed_thing(box, DirTree(self), flag=wx.EXPAND)

        # exclude some files?
        exclude, exclude_sizer = labeled_thing("Exclude some files by substring in filename?", wx.CheckBox(self, -1, ""))
        self.exclude = exclude

        box = wx.StaticBox(self, -1, "Exclude substrings...")
        self.exclude_list, self.exclude_boxsizer = exclude_list, exclude_boxsizer = boxed_thing(box, wx.TextCtrl(self, -1, "", size=(300,10), style=wx.TE_MULTILINE))
                                                                                                

        # Matching Mode
        mode, mode_sizer = labeled_thing("Identify channels by:", wx.Choice(self, -1, choices=match_modes))
        self.mode = mode

        # images notebook
        self.imagebook = imagebook = wx.Notebook(self, -1, style=wx.BK_TOP)
        imagebook.AddPage(ImagePage(self, imagebook, default_image_name(0)), default_image_name(0))
        imagebook.AddPage(wx.Panel(imagebook, -1), "Add another image...")

        # Layout
        self.otherdir_sizer = otherdir_sizer = wx.BoxSizer(wx.HORIZONTAL)
        otherdir_sizer.Add(otherdir_label)
        otherdir_sizer.Add(otherdir, 1)
        otherdir_sizer.AddSpacer(5)
        otherdir_sizer.Add(otherdir_browse)

        self.top_sizer = top_sizer = wx.BoxSizer(wx.VERTICAL)
        top_sizer.Add(base_dir_sizer, 0, wx.ALIGN_CENTER)
        top_sizer.AddSpacer(5)
        top_sizer.Add(otherdir_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, border=20)
        top_sizer.AddSpacer(10)
        top_sizer.Add(descend_sizer, 0, wx.ALIGN_CENTER)
        top_sizer.AddSpacer(5)
        top_sizer.Add(descend_update_filelist, 0, wx.ALIGN_CENTER)
        top_sizer.AddSpacer(5)
        top_sizer.Add(dirtree_boxsizer, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, border=20)
        top_sizer.AddSpacer(10)
        top_sizer.Add(exclude_sizer, 0,  wx.ALIGN_CENTER)
        top_sizer.AddSpacer(5)
        top_sizer.Add(exclude_boxsizer, 1, wx.ALIGN_CENTER)
        top_sizer.AddSpacer(10)
        top_sizer.Add(wx.StaticLine(self), 0,  wx.EXPAND | wx.LEFT | wx.RIGHT, border=20)
        top_sizer.AddSpacer(10)
        top_sizer.Add(mode_sizer, 0, wx.ALIGN_CENTER)
        top_sizer.AddSpacer(10)
        top_sizer.Add(imagebook, 3, wx.EXPAND | wx.LEFT | wx.RIGHT, border=20)

        border = wx.BoxSizer()
        border.Add(top_sizer, 1, wx.EXPAND | wx.ALL, 10)

        self.SetAutoLayout(True)
        self.SetSizer(border)
        self.Layout()

        # bindings
        base_dir.Bind(wx.EVT_CHOICE, self.change_basedir)
        otherdir.Bind(wx.EVT_TEXT, self.change_otherdir)
        descend_update_filelist.Bind(wx.EVT_BUTTON, self.start_update)
        otherdir_browse.Bind(wx.EVT_BUTTON, self.browse_otherdir)
        descend_dirs.Bind(wx.EVT_CHOICE, self.change_descend)
        exclude.Bind(wx.EVT_CHECKBOX, self.change_exclude)
        exclude_list.Bind(wx.EVT_TEXT, self.update_exclusions)
        mode.Bind(wx.EVT_CHOICE, self.change_mode)
        imagebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGING, self.page_changing)
        
        
        
        # default state
        base_dir.SetSelection(base_dir_choices.index(FS_DEFAULT_IMAGE))
        top_sizer.Hide(otherdir_sizer)
        descend_dirs.SetSelection(descend_dir_choices.index(FS_DESCEND_NO))
        top_sizer.Hide(descend_update_filelist)
        top_sizer.Hide(dirtree_boxsizer)
        exclude.SetValue(False)
        top_sizer.Hide(exclude_boxsizer)
        mode.SetSelection(match_modes.index(FS_SUBSTRING))

        self.update_file_list()
        self.Layout()
        self.SetAutoLayout(True)

    def change_basedir(self, evt):
        idx = self.base_dir.GetSelection()
        self.top_sizer.Show(self.otherdir_sizer, show=(base_dir_choices[idx] == FS_OTHER_DIR))
        if os.path.isdir(self.get_current_directory()):
            self.dirtree.set_directory(self.get_current_directory())
            if descend_dir_choices[self.descend_dirs.GetSelection()] == FS_DESCEND_NO:
                self.update_file_list()
        self.Layout()
        self.Refresh()

    def change_otherdir(self, evt):
        if os.path.isdir(evt.GetString()):
            self.dirtree.set_directory(self.get_current_directory())
            if descend_dir_choices[self.descend_dirs.GetSelection()] == FS_DESCEND_NO:
                self.update_file_list()

    def get_current_directory(self):
        idx = self.base_dir.GetSelection()
        return [default_input, default_output, self.otherdir.GetValue()][idx]

    def update_file_list(self, dir=None, descend_dirs=None):
        if dir is None:
            dir = self.get_current_directory()
        if descend_dirs is None:
            descend_dirs = descend_dir_choices[self.descend_dirs.GetSelection()]

        if descend_dirs == FS_DESCEND_NO:
            self.file_list = [(dir, '', f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        elif descend_dirs == FS_DESCEND_YES:
            self.file_list = []
            for dirpath, dirnames, filenames in os.walk(dir):
                subpath = relpath(dirpath, dir)
                print dirpath
                self.file_list += [(dir, subpath, f) for f in filenames]
        else:
            self.file_list = []
            for dirpath in self.dirtree.get_selected_dirs():
                subpath = relpath(dirpath, dir)
                self.file_list += [(dir, subpath, f) for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]

        self.file_list.sort()

        for i in range(self.imagebook.GetPageCount() - 1):
            self.imagebook.GetPage(i).update_file_list()

    def start_update(self, evt):
        self.update_file_list()

    def get_file_list(self):
        temp_list = self.file_list
        if self.exclude.GetValue():
            for exclude_substring in self.exclude_list.GetValue().split("\n"):
                if exclude_substring:
                    temp_list = [(d, s, f) for (d, s, f) in temp_list if exclude_substring not in f]
        return temp_list

    def browse_otherdir(self, evt):
        default = self.otherdir.GetValue() or default_input
        dlg = wx.DirDialog(self, "Choose a directory:", style=wx.DD_DEFAULT_STYLE, defaultPath=default)
        if dlg.ShowModal() == wx.ID_OK:
            self.otherdir.SetValue(dlg.GetPath())
            self.dirtree.set_directory(self.get_current_directory())
            # XXX update treectrl
        dlg.Destroy()

    def change_descend(self, evt):
        idx = self.descend_dirs.GetSelection()
        self.top_sizer.Show(self.descend_update_filelist, show=(descend_dir_choices[idx] != FS_DESCEND_NO))
        self.top_sizer.Show(self.dirtree_boxsizer, show=(descend_dir_choices[idx] == FS_DESCEND_CHOOSE))
        self.Layout()
        self.Refresh()

    def change_exclude(self, evt):
        self.top_sizer.Show(self.exclude_boxsizer, show=self.exclude.GetValue())
        self.update_exclusions()
        self.Layout()
        self.Refresh()

    def update_exclusions(self, evt=None):
        for i in range(self.imagebook.GetPageCount() - 1):
            self.imagebook.GetPage(i).update_file_list()

    def change_mode(self, evt):
        idx = self.mode.GetSelection()
        for i in range(self.imagebook.GetPageCount() - 1):
            self.imagebook.GetPage(i).set_mode(idx)

    def page_changing(self, evt):
        old = evt.GetOldSelection()
        new = evt.GetSelection()
        imagebook = self.imagebook
        if new == (imagebook.GetPageCount() - 1):
            imagebook.InsertPage(new, ImagePage(self, imagebook, default_image_name(new)), default_image_name(new))
            evt.Veto()

    def change_image_name(self, image_window, new_name):
        imagebook = self.imagebook
        for idx in range(imagebook.GetPageCount()):
            if imagebook.GetPage(idx) == image_window:
                imagebook.SetPageText(idx, new_name)

class ImagePage(wx.Panel):
    def __init__(self, file_selector, parent, initial_name):
        wx.Panel.__init__(self, parent, style=wx.BK_TOP)

        self.file_selector = file_selector

        # controls
        nlabel = wx.StaticText(self, -1, "Name for this image in CellProfiler: ")
        self.name = name = wx.TextCtrl(self, -1, initial_name)

        self.pattern_label = pattern_label = wx.StaticText(self, -1, pattern_label_strings[self.file_selector.mode.GetSelection()])
        self.pattern = pattern = wx.TextCtrl(self, -1, "")

        pathtext = wx.StaticText(self, -1, "Match against:")
        self.fullpath = fullpath = wx.Choice(self, -1, choices=match_elements)
        self.matches_only = matches_only = wx.CheckBox(self, -1, "Show only matching files?")

        self.file_list = file_list = wx.html.HtmlWindow(self, -1, style=wx.html.HW_SCROLLBAR_AUTO)
        if 'gtk2' in wx.PlatformInfo:
            file_list.SetStandardFonts()

        self.unprocessed_list = None

        # Layout
        name_sizer = wx.BoxSizer(wx.HORIZONTAL)
        name_sizer.Add(nlabel, 0)
        name_sizer.Add(name, 1)

        self.pattern_sizer = pattern_sizer = wx.BoxSizer(wx.HORIZONTAL)
        pattern_sizer.Add(pattern_label)
        pattern_sizer.Add(pattern, 1)

        self.path_sizer = path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        path_sizer.Add(pathtext, 0, wx.ALIGN_CENTER)
        path_sizer.AddSpacer(5)
        path_sizer.Add(fullpath, 0, wx.ALIGN_CENTER)
        path_sizer.AddSpacer(10)
        path_sizer.Add(matches_only, 0, wx.ALIGN_CENTER)

        self.top_sizer = top_sizer = wx.BoxSizer(wx.VERTICAL)
        top_sizer.Add(name_sizer, 0, wx.EXPAND)
        top_sizer.AddSpacer(10)
        top_sizer.Add(pattern_sizer, 0, wx.EXPAND)
        top_sizer.AddSpacer(10)
        top_sizer.Add(path_sizer, 0, wx.ALIGN_CENTER)
        top_sizer.AddSpacer(10)
        top_sizer.Add(file_list, 1, wx.EXPAND)

        border = wx.BoxSizer()
        border.Add(top_sizer, 1, wx.EXPAND | wx.ALL, 10)

        self.SetSizer(border)
        #bindings
        name.Bind(wx.EVT_TEXT, self.change_name)
        pattern.Bind(wx.EVT_TEXT, self.update_file_list)
        fullpath.Bind(wx.EVT_CHOICE, self.update_file_list)
        matches_only.Bind(wx.EVT_CHECKBOX, self.update_file_list)
        file_list.Bind(wx.EVT_SIZE, self.resize)
        file_list.Bind(wx.EVT_SCROLLWIN, self.scroll)
        file_list.Bind(EVT_CUSTOM_EVENT, self.restore_scroll)

        self.scroll_pos=0

        # initial state
        fullpath.SetSelection(match_elements.index(FS_FILENAME_ONLY))
        matches_only.SetValue(False)

        self.SetAutoLayout(True)
        self.Layout()

    def change_name(self, evt):
        self.file_selector.change_image_name(self, self.name.GetValue())

    def format_file(self, file_info):
        base_dir, sub_dir, filename = file_info
        pattern = self.pattern.GetValue()
        match_elements_choice = match_elements[self.fullpath.GetSelection()]
        modestr = match_modes[self.file_selector.mode.GetSelection()]
            
        prefix = ''
        if match_elements_choice == FS_FILENAME_ONLY:
            file_string = filename
            if sub_dir != '':
                prefix = '<font color="grey">' + sub_dir + os.sep + '</font>'
        elif match_elements_choice == FS_SUBDIR_AND_FNAME:
            file_string = os.path.join(sub_dir, filename)
        else:
            file_string = os.path.join(base_dir, sub_dir, filename)

        if modestr == FS_SUBSTRING:
            if pattern in file_string:
                return prefix + file_string.replace(pattern, '<font color="red">'+pattern+'</font>') + '<br>'
        elif modestr == FS_SHELL:
            if re.match(fnmatch.translate(pattern), file_string):
                return prefix + '<font color="red">'+file_string+'</font><br>'
        elif modestr == FS_RE:
            match = re.search(pattern, file_string)
            if match:
                start, end = match.start(), match.end()
                return prefix + file_string[:start] + '<font color="red">' + file_string[start:end] + '</font>' + file_string[end:] + '<br>'
        elif modestr == FS_POSITION:
            return prefix + '<font color="red">'+file_string+'</font><br>'

        if self.matches_only.GetValue():
            return ''
        else:
            return prefix + '<font color="grey">'+file_string+'</font><br>'

    def format_file_list(self):
        return "\n".join([self.format_file(f) for f in self.file_selector.get_file_list()])
        
    def update_file_list(self, evt=None, keep_pos=False):
        keep_position = keep_pos or ((evt is not None) and (evt.GetEventObject() != self.matches_only))
        try:
            if keep_position:
                self.file_list.Freeze()
                ux, uy = self.file_list.GetViewStart()
            self.file_list.SetPage(self.format_file_list())
            if keep_position:
                self.file_list.Scroll(ux, uy)
                self.file_list.Thaw()
        except re.error:
            # don't blow up when the pattern isn't valid
            pass

    def resize(self, evt):
        wx.PostEvent(self.file_list, MyEvent())
        evt.Skip()
        
    def scroll(self, evt):
        self.scroll_pos = self.file_list.GetViewStart()[1]
        evt.Skip()

    def restore_scroll(self, evt):
        self.file_list.Scroll(0, self.scroll_pos)

    def set_mode(self, idx):
        self.path_sizer.Show(self.fullpath, (match_modes[idx] != FS_POSITION))
        self.pattern_label.SetLabel(pattern_label_strings[idx])
        self.Layout()
        self.update_file_list(keep_pos=True)



class DirTree(CT.CustomTreeCtrl):
    def __init__(self, file_selector):
        self.file_selector = file_selector
        CT.CustomTreeCtrl.__init__(self, file_selector, -1, style=wx.TR_DEFAULT_STYLE)
        
        # folder images
        isz = (16, 16)
        il = wx.ImageList(*isz)
        self.fldridx     = il.Add(wx.ArtProvider_GetBitmap(wx.ART_FOLDER,      wx.ART_OTHER, isz))
        self.fldropenidx = il.Add(wx.ArtProvider_GetBitmap(wx.ART_FILE_OPEN,   wx.ART_OTHER, isz))
        self.SetImageList(il)
        
        self.set_directory(file_selector.get_current_directory())
        self.Bind(wx.EVT_TREE_ITEM_EXPANDING, self.expand)

    def set_directory(self, dir):
        if not os.path.isdir(dir):
            return
        self.DeleteAllItems()
        root = self.AddRoot(dir, ct_type=1)
        self.SetPyData(root, (dir, False))
        self.SetItemImage(root, self.fldridx, wx.TreeItemIcon_Normal)
        self.SetItemImage(root, self.fldropenidx, wx.TreeItemIcon_Expanded)
        self.AppendItem(root, '...')
        self.Refresh()
        
    def expand(self, evt):
        self.Freeze()
        item = evt.GetItem()
        (dirname, already_visited) = self.GetPyData(item)
        if not already_visited:
            # put some feedback about directory size here?
            subdirs = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]
            subdirs.sort()
            if len(subdirs) > 0:
                self.DeleteChildren(item)
                for sd in subdirs:
                    child = self.AppendItem(item, sd, ct_type=1)
                    self.SetPyData(child, (os.path.join(dirname, sd), False))
                    self.SetItemImage(child, self.fldridx, wx.TreeItemIcon_Normal)
                    self.SetItemImage(child, self.fldropenidx, wx.TreeItemIcon_Expanded)
                    self.AppendItem(child, '...')
        self.SetPyData(item, (dirname, True))
        self.Thaw()

    def get_selected_dirs(self):
        def find_all_checked_branches(node):
            if node.IsChecked():
                yield self.GetPyData(node)[0]
            for child in node.GetChildren():
                for checked_child in find_all_checked_branches(child):
                    yield checked_child

        return [d for d in find_all_checked_branches(self.GetRootItem())]


class MyApp(wx.App):
    def OnInit(self):
        frame = CPFileSelector(None, title="Select files to load...")
        frame.Show(True)
        self.SetTopWindow(frame)
        return True

app = MyApp(0)
app.MainLoop()
