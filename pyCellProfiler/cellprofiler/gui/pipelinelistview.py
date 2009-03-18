"""PipelineListView.py

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import StringIO
import wx
import wx.grid

import cellprofiler.pipeline
import cellprofiler.gui.movieslider as cpgmov
import cellprofiler.gui.cpgrid as cpgrid

NO_PIPELINE_LOADED = 'No pipeline loaded'
PADDING = 1

IMG_OK = '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\
\x08\x06\x00\x00\x00\x1f\xf3\xffa\x00\x00\x00\x06bKGD\x00\xff\x00\xff\x00\xff\
\xa0\xbd\xa7\x93\x00\x00\x00\tpHYs\x00\x00\x00H\x00\x00\x00H\x00F\xc9k>\x00\
\x00\x00\tvpAg\x00\x00\x00\x10\x00\x00\x00\x10\x00\\\xc6\xad\xc3\x00\x00\
\x02|IDAT8\xcb\x9d\x93]HSq\x18\xc6\x9f\xb3\xf3\xb13g\xd3\xb9\x9a\xba9\xdd,\
\x17\xd3\x14\xd1\x90%I(\x81\x82&\x91\xddhPwQfE\x08\x06A(QA]\x18t\xd3\x85\x08v\
\xa5\tEDTWz\x11b\x82\x85$\xb9\xd2Dk\xb59k\xeb,\xcf\xce\xd9\xce9\xdb\xbf\x0b\
\xa5\x08\x93V\xcf\xed\xcb\xef\xfdx\x1e^\n\xff\xa7\x0c\xce\xc1_\xe0XRF\xff\
\x13FQ\x00`6\x95\x19\xfaJ;L\xddt"iI\x9bex\n\x00\x1cEu\xfc\xf0\xe1\xbby\xc9\
\xfak\x16\xc1`\xd1\x1dO\x0b\xce/\xa6\xc0\xd0\xa8\xa8>\xc2\x8d\x9f|j\'\xed\
\xf7]\xc4Z\xc9\xf7\x03`\xd3j`\xcaBC\xd3)v\xb6g\xd2A:\xa7\xca\xc9\xee\xb6\xecI\
\x00\x05:f\xbd\xbe\r@\xe6\x16,m\xb5\xa3\xbd\xe32\xeb\xef{\xed"\xdds5\xa4\xf6\
\x92\xed+c\xa4\x9aL\xae\xf5\xe1\xb4\xd3\xcb\x8d8\xf7\xeaN\xa8B*&\x8b\xf8\x00@\
\xdd\x80y\x9b\x9b:\xd7x\xd6p\xc3\xdd\xe6\xb4\x8a\xb4\x15o\xa7\xc2\xe4\xd5@\
\xb0_\n&\x07\x13B\x8a\x00\x00}\xa0\x95\xb9s\xfa\x8a\xb5\xd4U\x1ao\x91\x95TE\
\xe8\x13\x15L\xa9\x90m\x95L\xef\xc1\xf3\xa6\x8b;\x1b]\x99b\xca\x8c@@\xc4\xf4\
\xc0\xe7\xb1\x95\x17R\x0f\xa5C\x0cdc\xc5\x84\x942Wy=^oc\xb9>\xbfJ\xf6$\xb2I\
\x8b\xc8q\x87\x1a:\xcd\xad\xbb\xf6\x17\xe9c\xaa\x11\x119\x8e\x99\xd1`p\xfe^\
\xa4\x8b\xb7\xd0>-F~\xdd(|\x81O\xd1I\xb5\x9e\x9a\x12\x87)\xa7\x10\xb9e\xb4\
\xd1\xbe\x8f\xb3\xd9\x8bsiI\xe5\x10\xd5T,L\x84\xb57C\xab\xd7\x95prX\x93\xc8\
\xef&9\xab\xe8\xb5\xf9ii-\xb3Pk\xce-\xd9\xce*I=\x18\x8e\x87\xa8\x00QUE\
\xc0/bvp\xe5qdF\xbe\x0c\n\xf1M.\x0bA\x02E\xc2\x92(\xcb\x1ek%\xbf\'\x95A#\
\x9aP (*"R\x02s#\xa1\xe5\x8f\x0f\xbe\x9daM\xf4R*A6\xc5\xa4\x03\x00>\x8b\
\x92\xdf\x8d\xcb\xb7&\x1e\xf9W"\xb2\x84\xa8\xa2"\xaajX~.(\xfe\'\xc2\xcd\
\xe6\xc5\xf2\x97\xea\xf7\xe4\x9fs\x06\x00-\x01\x80  \nj\x8e\xc1\xcd\xd5\
\x11\x0b\x8b\xe0b\x0c\xf3C\xa1\xd15_\xfc\xea\xc2\xedU\x15[\xe8\xe73Q\x0c\
\x05)\xa4\xbd\xd7X\xd4S\xf9l\xde\xd2\xc3\xb0/\xf4L\xe8\xa2\r\xba\x00\xd1\
\x08\xd2R\xe1\xb1\x1c\xe8\xadL[v\xb5q\x8c\xdb\xc1\x1eu\xf7\x16\xfc\x95\
\xf9\x01\x99\xb1\x05I\xe6\xe0\xf19\x00\x00\x00%tEXtcreate-date\x002008-10-18\
T18:46:24+08:00\xa3\xb5\xdd\xcb\x00\x00\x00%tEXtmodify-date\x002008-10-18\
T18:46:24+08:00\xfc\x04\xab\xff\x00\x00\x00\x00IEND\xaeB`\x82'

IMG_ERROR = '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\
\x10\x08\x06\x00\x00\x00\x1f\xf3\xffa\x00\x00\x00\x04sBIT\x08\x08\x08\x08|\
\x08d\x88\x00\x00\x00\tpHYs\x00\x00\x01\xb4\x00\x00\x01\xb4\x01L"M\xf8\x00\
\x00\x00\x19tEXtSoftware\x00www.inkscape.org\x9b\xee<\x1a\x00\x00\x02~IDAT8\
\x8d}\x93KHTa\x14\xc7\x7f\xf7\xce\\\xc7\xb93\x96\x9a\x92=L\xe9AS)\t\xc1,\x8aB\
\xa1E\x8b\x82\x10"j!\x13)\xd5B\x82v-\x8a\xa2]\x0b\x89\x16J\xa8Y\x9aI\xdb\x82\
\xa0\xa8\xc6Md-\\\x8496b\x19\xf4\x98f\xc6\xb2h\x9c{\xaf\xcewZ\xcc\xa3\xc4\xf2\
\x0fgq\xce\xe1w\x1e\x1f\xdf\xd1D\x84\xe1"\x9fWs\xb9.{7n<\xec^U\xbe\t\x11\x1d\
\xa5\x90\x9c\x91Q\x88\xca \x99\x8c\xb2\xa6?\xbcU\xe9\xf4\x03\xe0R\xa3\x93Jka\
\xc3\xf4z\xaa\xabG\xab\xcf\xb6\x07\x8ak\xd6\xa3,\x1be;(\xdb\xce\x9ae#\xb6\
\x8drr~*\xc5\xec\xf3\x17\xa4\xc6\xde\x8c\x01A\xb7f\x18\x17\xd7\x9dj\rx7T\xe1\
\xdb\xbe\x15\'\x9e\xe0\xc7\xcbQ\x94ee!\xcbB\xd9\x0e\xe28\x945\xee\xc6S\xb5\
\x1a]\xd2\xcc\x8dG\xeaD\xa9\x0bnomm\xb3\xcb\xef\xc3SSC\xf1\x96\x00\xc5[\x02\
\xe8>\x93\xd8\xc0\xbd\x1cl#\x8e\xc3\xda\xd6\x16V\xeek\x02\xc0\x9a\x9e\x82\
\x9eA\x80C\xba\xee37;\x89$\xd6\xbb\xf7\xe4\xe5o\xd8Ee\xf3A\xe6"\x13|\x7f\
\xfc\x84\xca\x03M\x05\x18\xe0\xe7\x8b\x11D\x01\xb0\xcd-\xb6\xa3\xcf\'\x92$\
\x1f=\xc5]^Fi\xd3~\x00V\xee\xd9\xc7\x9a\xd0Q4\xa0\xe2xK\x01\x8e\xf5t\xf1\xb1w(\
\xef\xba\xb4\x91\xad;\xc4\xdf\xd0\x80\xb2\xb3\xe3\xae;q\x9c\x8a#\xc7\xf8\
\x97b7o0q\xe6\x1c\xc8\x9f\x98[Y\x16\xf3\xc9$\xca\xb2\x98\x8bN2\xf6\xf4\x19\
\x81\xd4/\xaaBm\x8b\xe0\xaf\xb7\xba\x97\xc0\xb9\x026\xf3\x89\x04N"\xc9\xc2\
\xec,h\xffl\xfe_\xe9\xca\xb2p\xe2\t\xe6gf@\x83m]\x1dK\xba\x03\xac\x0e\xb5\
\x11\xe8\xeaX\xda`\xb8\xb8D\xc2\x86)\xe1"S>ww\xca\xdf\x8a\xdd\xee\x91\xaf\
\xb7{\x16\xc5\xbetwJ\xb8\xc8\xcc2\x86)\x84\rs!l\x98\x12=}r1|\xa7O\x86=~\x19\
\xf6\xf8%>\xd0\xb7(\x17m\x0b\xe5\x0b,\xe8@D\xd3\xa1\xa4\xa1\xbe0U|\xa8\x9f\
\x89\x93\xed\x85[\x18om\'q\xb7\xbf\x90\xf7\xed\xac\xcf\xaf\x12q\x85\\F5\xc2^+\
\xfa\x1a\xa3t\x05\xf1\xc1\x01\xa6\xce_\xc9\x1eQ^"$\xef?Dfc\xd8\xd3\x93|\xba~\r\
\xe7\xdb\x1c@\xaf\x166L/\xf0\n\xa8\xd3\\ \x99\xe5_]\xd3\xc9\xff\xc21 \
\xa87:\xa94\x10\x04\xaeJ\x86q@-\xc3+Q\x8c\x03W\x81`\xa3\x93J\xff\x06\\\xf6u2\
\x18<\xfe\xa2\x00\x00\x00\x00IEND\xaeB`\x82'

IMG_EYE = '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00 \x00\x00\x00 \x08\
\x02\x00\x00\x00\xfc\x18\xed\xa3\x00\x00\x03\xd9IDATx\x9cbd`h`\xa0%\x00\x00\
\x00\x00\xff\xffb``h\xf8\x8f\x03\x1c8p\xa0\xa1\x01\xbb\xf5\r\r\r\x07\x0e\x1c\
\xc0\xa5\x11\x0e\x18\x18\x1a\x00\x00\x00\x00\xff\xff\xc2n\x01.sq\xba\xb1\x01\
\xbb+\x19\x18\x1a\x00\x00\x00\x00\xff\xffB\xb7\xa0\xa1\xa1\x81T\xd3\xf1\xd8\
\xc4\xc0\xd0\x00\x00\x00\x00\xff\xffbd`h\xf8\xff\xbf\x1e"\xdd\xd8\xd8\x88f:\
\x9c\xeb\xe0\xe0p\xe0\xc0\x014A\\\xe0\xff\xff\xff\x10\x06#c#\x00\x00\x00\xff\
\xffB\xf8\x00Y\x1b./C"\x06\x127x\xac\x81kg`h\x00\x00\x00\x00\xff\xffBX@\xd0h\
\x88\xe9\xc8$<\x15\xa0\x91\xc8\x16\x00\x00\x00\x00\xff\xff\x82Z\x80,\x8al\x16\
.s\xd1\xd8\x98v\xc0-\x00\x00\x00\x00\xff\xffb\x84\x07i}}=\x9aO\x0f\x1e<hoo\x7f\
\xf0\xe0A\x06\x06\x068\xe3\xc0\x81\x03\xc8\xf1\xe1\xe0\xe0\x00W\x8f,hoo\x0f\
\x89\x03\x00\x00\x00\x00\xff\xffB\x89dds\xe1lLC\xe1\x81\x89\x9c\x04\x90\xa5 \
\xae\xb1\xb7\xb7gdl\x04\x00\x00\x00\xff\xffB\xb1\x00n4\xc4\\4w\xc1\rEK\x0e\xb8\
\xec\x80\xf8\x00\x00\x00\x00\xff\xff\x82Z\x80i4\xb2\xab\x1d\x1c\x1c\x1c\x1c\
\x1c\xe0\t\x01Y\rV\x7f \xfb\x00\x00\x00\x00\xff\xffB\xf8\x00Y\x1b\xdcu\x07\x0e\
\x1c\x80\xb9\x85\x11n\x01D1Z\x88a\xda\x01\xf1\x01\x00\x00\x00\xff\xffb\xc1j4\
\x9a:\x08\x80\xd8\x81i4D1\x9a <H\x00\x00\x00\x00\xff\xffb\xc1T\x8a\x1c\x1a\x8c\
\x8c\x8c\xc8\x81\xc0\xc8\xc8\x88\xe9\xea\x86\x86\x06H\x18B\x92,$\t\xc1\x1d\r\
\x00\x00\x00\xff\xffbd`h8p\x00\xcd_XL\xc1\x0f v0\xc0b\x8e\x01\x16\r\x0e\x0e\
\x07\x00\x00\x00\x00\xff\xffbBv~cc#\x03\xf6\xa2\x06\x92\'\x91\xd1\x7f\x06\x06D\
\x94@b\x8b\x01\xa9\xc8\x82\x07\x11\x00\x00\x00\xff\xffb``@\x94\xec\x98\xeeb`\
\xf8\xcf\xd0\xd0\xc0\x00K0H\xe2\x0c\x0c\r\r\x0c\x0c\xe8%\x18\xbc\xfc\x80\xe7d\
\x00\x00\x00\x00\xff\xffb\x80\x14\x15\x98F\xc3\x1c\xf5\x9f\xa1\xa1\x81\x01Q\
\xfc"T@,@\xab\x94\xe0v@\x1c\xcd\xc0\xd0\x00\x00\x00\x00\xff\xff\x82\x06\x11rM\
\x00!\xa1\xd9\xbd\xa1\x91\x81\x81\x81\xa1\xb1\xf1\xe0\xc1\x83\x90\x00D\t\xc3\
\x86F{{{\x07\x07\x07d\xcf\xc1K\'\x08\x00\x00\x00\x00\xff\xffB\xa4"\xe4\xb2\x08\
^\x9804\xd4C\xecp8p\xa0\x01\xee9\x84\x83\xeb\x19\xeaQ\x922\xa6\x1d\x00\x00\x00\
\x00\xff\xffBD\xf2\xc1\x83\x07!i\x0ba:\x03\x03CC#C\x03\xd4b\x84\xef\xe0\xa6742\
\xc0\xb2\x11\xb2\x0c<\xd5200\x00\x00\x00\x00\xff\xffBX\x00\xcf\xb4\x0cHY\xef\
\x7f}=\xb2\x1dH\xee\xacghh\xfc__\xcf\x00K\xdc\xf5\xf5\xf5\xf0p\x86x\x02\x92^\
\x01\x00\x00\x00\xff\xffb\x80D2,N\xe0\xd1\x89R\x1f044004\xa0\x91\x98*\x19Pk\
\x1bH$\x03\x00\x00\x00\xff\xffB\x94E\x8c\x8c\x8c\x10O\xa0\x15;p\xc0\x08\x8b\
\xe4\xff\xa85\x07\xbcH\x87G\x00<$\x18\x19\x1b\x01\x00\x00\x00\xff\xffb\x80W\
\x99pO \xbb\x08O\xf5\tw2D\x0br\x1d\x87\xec\x03\x00\x00\x00\x00\xff\xffBoU\xc0\
\x03\x0e^\xb6\xa0U\x00\xf0p\x80\xb3!\xc1\r)\x07\xd1R\x14#c#\x00\x00\x00\xff\
\xffbB\xe6\xd7\xd7\xd7\xdb\xdb\xdb\xa3\xd5\xb1\xc8f\xa1\xd9\x04/v\x0e\x1c8\
\xd0\xd8\xd8\x88f:\x04\x00\x00\x00\x00\xff\xffbB\xe3766B2\x04D3\xbc\x14C\xe6\
\xc2\x05!a\x081\x1a\xb3J\x87\x00\x00\x00\x00\x00\xff\xffb\xc0\xdatDkF@\xc2\
\xa1\x01\xd6\x1e\x85\x90\x90\xd4r\x00o\x0b\x95\x81\xa1\x01\x00\x00\x00\xff\
\xffb\xc1j\'\xb2g!ld\x12-|p9\x1c\x02\x00\x00\x00\x00\xff\xffb\xa4u\xf3\x1d\
\x00\x00\x00\xff\xff\x03\x00\xb2\x93\xdaw\x985.[\x00\x00\x00\x00IEND\xaeB`\x82'

ERROR_COLUMN = 0
EYE_COLUMN = 1
MODULE_NAME_COLUMN = 2
ERROR = "error"
OK = "ok"
EYE = "eye"

class PipelineListView(object):
    """View on a set of modules
    
    """
    def __init__(self,panel):
        self.__panel=panel
        self.__sizer=wx.BoxSizer(wx.HORIZONTAL)
        self.__panel.SetSizer(self.__sizer)
        self.__pipeline_movie = cpgmov.MovieSlider(self.__panel,
                                                   value=0,
                                                   min_value=0,
                                                   max_value=1,
                                                   start_value=0,
                                                   stop_value=1)
        self.__sizer.Add(self.__pipeline_movie,.5,wx.EXPAND)
        grid = self.__grid = wx.grid.Grid(self.__panel)
        self.__sizer.Add(self.__grid,1,wx.EXPAND)
        grid.CreateGrid(0,3)
        grid.SetColLabelSize(0)
        grid.SetRowLabelSize(0)
        error_bitmap = wx.BitmapFromImage(wx.ImageFromStream(StringIO.StringIO(IMG_ERROR)))
        ok_bitmap    = wx.BitmapFromImage(wx.ImageFromStream(StringIO.StringIO(IMG_OK)))
        eye_bitmap   = wx.BitmapFromImage(wx.ImageFromStream(StringIO.StringIO(IMG_EYE)))
        error_dictionary = {ERROR:error_bitmap, OK:ok_bitmap}
        eye_dictionary   = {EYE:eye_bitmap}
        cpgrid.hook_grid_button_column(grid, ERROR_COLUMN, 
                                       error_dictionary, hook_events=False)
        cpgrid.hook_grid_button_column(grid, EYE_COLUMN, eye_dictionary,
                                       hook_events=False)
        name_attrs = wx.grid.GridCellAttr()
        name_attrs.SetReadOnly(True)
        grid.SetColAttr(MODULE_NAME_COLUMN, name_attrs)
        grid.Bind(wx.grid.EVT_GRID_CELL_LEFT_CLICK, 
                  self.__on_grid_left_click, grid)
        grid.Bind(cpgrid.EVT_GRID_BUTTON, self.__on_grid_button, grid)
        wx.EVT_IDLE(panel,self.on_idle)
        
    def __set_min_width(self):
        """Make the minimum width of the panel be the best width
           of the grid and movie slider
        """
        text_width = 0
        dc = wx.ClientDC(self.__grid.GridWindow)
        for i in range(self.__grid.NumberRows):
            font = self.__grid.GetCellFont(i, MODULE_NAME_COLUMN)
            text = self.__grid.GetCellValue(i, MODULE_NAME_COLUMN)
            text_width = max(text_width,dc.GetFullTextExtent(text, font)[0])
        self.__grid.SetColSize(MODULE_NAME_COLUMN, text_width+5)
        self.__panel.GetTopLevelParent().Layout()

    def attach_to_pipeline(self,pipeline,controller):
        """Attach the viewer to the pipeline to allow it to listen for changes
        
        """
        self.__pipeline =pipeline
        pipeline.add_listener(self.notify)
        controller.attach_to_pipeline_list_view(self,self.__pipeline_movie)
        
    def attach_to_module_view(self, module_view):
        self.__module_view = module_view
        module_view.add_listener(self.__on_setting_changed_event)
        
    def notify(self,pipeline,event):
        """Pipeline event notifications come through here
        
        """
        if isinstance(event,cellprofiler.pipeline.PipelineLoadedEvent):
            self.__on_pipeline_loaded(pipeline,event)
        elif isinstance(event,cellprofiler.pipeline.ModuleAddedPipelineEvent):
            self.__on_module_added(pipeline,event)
        elif isinstance(event,cellprofiler.pipeline.ModuleMovedPipelineEvent):
            self.__on_module_moved(pipeline,event)
        elif isinstance(event,cellprofiler.pipeline.ModuleRemovedPipelineEvent):
            self.__on_module_removed(pipeline,event)
        elif isinstance(event,cellprofiler.pipeline.PipelineClearedEvent):
            self.__on_pipeline_cleared(pipeline, event)
    
    def select_one_module(self, module_num):
        """Select only the given module number in the list box"""
        self.__grid.SelectBlock(module_num-1, MODULE_NAME_COLUMN,
                                module_num-1, MODULE_NAME_COLUMN,
                                False)
        self.__on_item_selected(None)
        
    def select_module(self,module_num,selected=True):
        """Select the given one-based module number in the list
        This is mostly for testing
        """
        self.__grid.SelectBlock(module_num-1, MODULE_NAME_COLUMN, 
                                module_num-1, MODULE_NAME_COLUMN,
                                True)
        self.__on_item_selected(None)
        
    def get_selected_modules(self):
        return [self.__pipeline.modules()[i]
                for i in range(self.__grid.NumberRows) 
                if (self.__grid.GetCellValue(i,MODULE_NAME_COLUMN) != 
                    NO_PIPELINE_LOADED and
                    self.__grid.IsInSelection(i, MODULE_NAME_COLUMN))]
    
    def __on_grid_left_click(self, event):
        if event.Col == MODULE_NAME_COLUMN:
            self.select_one_module(event.Row+1)
            #event.Skip()
    
    def __on_grid_button(self, event):
        pass
    
    def __on_pipeline_loaded(self,pipeline,event):
        """Repopulate the list view after the pipeline loads
        
        """
        nrows = len(pipeline.modules())
        if nrows > self.__grid.NumberRows:
            self.__grid.AppendRows(nrows-self.__grid.NumberRows)
        elif nrows < self.__grid.NumberRows:
            self.__grid.DeleteRows(0,self.__grid.NumberRows - nrows)
        
        for module in pipeline.modules():
            self.__populate_row(module)
        self.__adjust_rows()
    
    def __adjust_rows(self):
        """Adjust movie slider and dimensions after adding or removing rows"""
        self.__set_min_width()
        self.__pipeline_movie.slider.max_value = self.__grid.NumberRows
        self.__pipeline_movie.slider.value_names = \
            [module.module_name for module in self.__pipeline.modules()]
    
    def __populate_row(self, module):
        """Populate a row in the grid with a module."""
        row = module.module_num-1
        self.__grid.SetCellValue(row,ERROR_COLUMN, OK)
        self.__grid.SetCellValue(row,MODULE_NAME_COLUMN, 
                                 module.module_name)
        
    def __on_pipeline_cleared(self,pipeline,event):
        self.__grid.DeleteRows(0,self.__grid.NumberRows)
        self.__adjust_rows()
        
    def __on_module_added(self,pipeline,event):
        module=pipeline.modules()[event.module_num-1]
        if (self.__grid.NumberRows == 1 and 
            self.__grid.GetCellValue(0,MODULE_NAME_COLUMN)==NO_PIPELINE_LOADED):
            self.__grid.DeleteRows(0,1)
        self.__grid.InsertRows(event.module_num-1)
        self.__populate_row(module)
        self.__adjust_rows()
    
    def __on_module_removed(self,pipeline,event):
        self.__grid.DeleteRows(event.module_num-1,1)
        self.__adjust_rows()
        self.__module_view.clear_selection()
        
    def __on_module_moved(self,pipeline,event):
        module=pipeline.modules()[event.module_num-1]
        selected = module in self.get_selected_modules()
        if event.direction == cellprofiler.pipeline.DIRECTION_UP:
            other_module = pipeline.modules()[event.module_num-2]
        else:
            other_module = pipeline.modules()[event.module_num]
        self.__populate_row(module)
        self.__populate_row(other_module)
        if selected:
            self.__grid.SelectBlock(module.module_num-1, MODULE_NAME_COLUMN,
                                    module.module_num-1, MODULE_NAME_COLUMN,
                                    False)
        self.__adjust_rows()
    
    def __on_item_selected(self,event):
        if self.__module_view:
            selections = self.get_selected_modules()
            if len(selections):
                self.__module_view.set_selection(selections[0].module_num)
    
    def __on_setting_changed_event(self, caller, event):
        """Handle a setting change
        
        The debugging viewer needs to rewind to rerun a module after a change
        """
        setting = event.get_setting()
        for module in self.__pipeline.modules():
            for module_setting in module.settings():
                if setting is module_setting:
                    self.__pipeline_movie.slider.value = module.module_num - 1
                    return

    def on_idle(self,event):
        modules = self.__pipeline.modules()
        for idx,module in zip(range(len(modules)),modules):
            try:
                module.test_valid(self.__pipeline)
                target_name = module.module_name
                ec_value = OK
            except:
                ec_value = ERROR
            if ec_value != self.__grid.GetCellValue(idx,ERROR_COLUMN):
                self.__grid.SetCellValue(idx,ERROR_COLUMN,ec_value)
