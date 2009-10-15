"""PipelineListView.py

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

from StringIO import StringIO
import base64
import zlib
import wx
import wx.grid

import cellprofiler.pipeline
import cellprofiler.gui.movieslider as cpgmov
import cellprofiler.gui.cpgrid as cpgrid

NO_PIPELINE_LOADED = 'No pipeline loaded'
PADDING = 1

def plv_get_bitmap(data):
    raw_data = zlib.decompress(base64.b64decode(data))
    return wx.BitmapFromImage(wx.ImageFromStream(StringIO(raw_data)))

IMG_OK = ('eJzrDPBz5+WS4mJgYOD19HAJAtICIMzBBiTlP/9PBFJsSd7uLgz/QXDB3uWT'
          'gSKcBR6RxUDaA4zdTmbbgQTLChzTYfoZYo6tPczAwFTj6eIYYnF67uRYj+BC'
          'iWPzN3/eaJx+eeesXZZ3dcQvi1ycoOqp0dikNvFuRkB5YJorB5ujRqBjrETJ'
          '5VaOsqWcri4h4VWCSU2tKjsvuWRvtcx+rXP+3M1zlrf3cy/lmBx2/u3p938r'
          '5OK4/i/nOXcw/kFEkNt/YbdAhgSzqZK/vKx97pYoZXrOTq3gYpBxLf3z4eHu'
          'ypO/ssUOJlyU9ec+p7/swIUVK+wO9c6vyVJ/+z32SNTJ78wJl7MSTjk5X9Ys'
          '25Z+ydFq+amT77a98WRgtUrbu4/X4ZmYTu7WxXsfG71+X/1W6W6x6ZJvk95q'
          'Jy+Z5bPu68Mtl0/3Wnx/5bfCSUut+weDw92Gytmzra5XXCs4fPfZlq4tovnL'
          'Dz256rAhnkuNXdipi4GhdsHUncW/urZeCZXKnzg1xPWFsKjPygm5U33eH/y8'
          'rNtaOnZm0qkeB4cjXw483zhVPIh/qTNPSvLRlilG4ZW2cfnJO+32e31T2eTZ'
          'faKwXcrq7Nrd38RfJq8SFLTsm3kxoeBf3JLu7RfsdN3q7mrUNPpf9Nw6b5ZQ'
          'u+ZygZ2pWy7u6998s7s40/OpwNUQHZ+WrebOq69PLSiKmHzivZrl6hdbf2Zm'
          '6m4OyD6ne/OclqetRF/7CobAUNcD+kllBU8LU9z28XB99NXjdmRyPTRJ47Rc'
          'tup+9amOyrMCFDS0lIKYipUXPu3n3zc30fdLkJaj2dElzAx23ZPu957erib3'
          'M1xpU8uKRUqrsiLq9DT+qR86++zop+mvvj+ZX8zGoMvYoKDAldV38OxVQe7u'
          'B0k8n50XXjSN//Pq0NtQ0egXz40DeVg1l+y9HnEl+GfOvUuHN+h/8XmxiHcX'
          'w0WOS0EPN8q8WOsTXba1sOf2QbnS72J/pv5knLmR1fPZg4+WwASpWuIaUZJc'
          'lJpYkqqbAiQYjAwMLHQNDXQNLUIMLaxMzKyMTLQNLKwMDBZvvXsapiE3PyUz'
          'rZKAhj8sq/8DNTB4uvq5rHNKaAIAzkFukQ==')

IMG_ERROR = ('eJwBSQK2/YlQTkcNChoKAAAADUlIRFIAAAAQAAAAEAgGAAAAH/P/YQAAAAlwSFlz'
             'AAALEwAACxMBAJqcGAAAAAd0SU1FB9kHGxUkE6NmKioAAAHoSURBVDjLlZPB'
             'ShthFIW/O5nONIkzgxCZSu1GQ8QghkFKmjAEgkhjg3QXuhAiXXTnMBvxCVwb'
             'dGXAQsBFSVYFra9QCD6AUFwECoUWXNRASTF/N9YSyLT2wL2Ly+E7Z3OFCIXw'
             'yIC3AAN43YAv43x6FMCEo3l4DnABR0B1nE+LSPdSsFpYWNCepFJaClZD8O4N'
             'MGEvDXpid5dMs0kadBP27gUIoTgFxWy5LHo+j57Pky2XZQqKIRT5l3agewLD'
             'XqejPM9TnuepXqejTmC4A92/NghhxYWljO+L4fs4joPjOBi+T8b3xYWlEFYi'
             'ASbsZ8FIBAGiaViWhWVZiKaRCAKyYJiwPxYQwvo0pOcKBYxSCRHBtm1s20ZE'
             'MEol5goFpiEdwvoIIAQxoLF4m67pOiLyp4EImq6TCAIWwTCgEYLA7Qrh1Sy0'
             'XiwvG5NnZxCLoZSi3+8DkEwmERG4ueFqbY0P5+eDS6g34J2EEHsMV0/Bmj8+'
             '5mG1ilIKpRS1Wg2AdruNiCAi/Dg95WJjgy58/wyTmsBmHMyZXI54pXJnFBFc'
             '18V13ZFbvFJhJpcjDqbApmzD15eQmj04YKJev0v/PcAIQES4brW43NriPXyT'
             'Qxg+A3nA/+kn8BGU/gkOB/BmIuIvonQNwx40fwGX3n0IHLThEwAAAABJRU5E'
             'rkJggtPN7j0=')

IMG_EYE = ('eJwBCQP2/IlQTkcNChoKAAAADUlIRFIAAAAQAAAAEAgCAAAAkJFoNgAAAAFz'
           'UkdCAK7OHOkAAAAEZ0FNQQAAsY8L/GEFAAAAIGNIUk0AAHomAACAhAAA+gAA'
           'AIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAKHSURBVDhPbVJfSFphFD+DGpcx'
           'oocYIXsIJxLSQw+xeuwhzIc9xQiRIREi9yGW2L9LifMhnQwpqSGSI4aN8EFk'
           'RLSl2xCDdlcjb4MkY5MMKxxuRqjVctz9bneDMXb4+Djf9/1+5/zOd8611lZ/'
           'd/edapXq6upqa2v9fv/BQYaohqhiNLLt7e2VSqV0ZTU1tLr6hTjujXhlXq+b'
           '/mf19cz09GMZAzAND0sEGel0Ond2dvL5fDqddrlcbW1tf0cADGCanPxoMj0I'
           'BAI4X1xcyHuhUMhkMvAFQejq6uro6GhpaREEnuPipNN5y+VveCsWi5eXl0Cs'
           'r6/7fD6PxwMoSsIxHo/Pz8+fn59IGUASxZ+5XC6VSi0vL9tsNoPBsLCwAIJe'
           'r1cqlRzHRaPRRCJxevpVIphM4f39XTwjmEajQSpwstkssgWDwZmZmaWlJZZl'
           'NzY2dncFSVJPzxzPI4moVqu1Wu2N6//+lN1u53ne6/VWqyWjcVEibG6u9fb2'
           '3oQxRLf676oXFQpzs2qAGt8rFf0IgJwoA3HHxxPkdgs6XSduofgk95l65nxi'
           'ebTfcBtXc+Ou4KnDNixzEomow8GT1RqFHrksqSFMnJ48bNLfb/Z76N4zzn/8'
           'o/JdVul2Ozo73TQ6+g5FT01NSd96dpYVD4l5SnqOdANNrNRT9AR9RCXb2x9Y'
           '9iVBVij0PBKJhEKhWCwmj8Cj5ObxlTM7OwsoPspqteIIOWSxvIaXTCbD4bDF'
           'YmEYBnI1SmVjQwM6oFKp0LKtrS05EMC/hy+fz5ZKhaOjzN7ep3D4hdlshL+y'
           'EhHFarEoJ/szfBjvsbHY0FBsZORtX19oYmINQhHJbI5g4RJPg4OvsAAD+BeM'
           'WKsmyxjGwwAAAABJRU5ErkJggi0SXAI=')

IMG_CLOSED_EYE = (
    'eJwBsgJN/YlQTkcNChoKAAAADUlIRFIAAAAQAAAAEAgCAAAAkJFoNgAAAARn'
    'QU1BAACxjwv8YQUAAAJpSURBVDhPbVJBSCJhFP4XSsSDePYUrMeIDkHSqUOU'
    'Jw+dPCxDBxHpkljZUBESkyuL1CAiskJIyDKUiIS4trlLmMRsDTYZwRSstBEt'
    'bi4uUaLpMvuNs8iy7ONnePO/73vve+9/LwYHIxMTL9ttotfre3t7I5HIzU2Z'
    'kB5C6hTlHB4ertfrjx3r6SF7e18ITefkjrGsn/zPDAbtxsZrFQMwmZtTCCpy'
    'bW3t4uKiUqlcXl76fL6hoaG/MwAGMGEYwW5/FY1G8d9sNtVvtVotl8vwRVEc'
    'Gxszm839/f2iyNP0AbFY2KenH4jVarVWqwXE0dFROBwOBAKABoPBg45tbm42'
    'Gj+VCiDJ8q/b21tJkra3t6HEZrPt7OwAarVaKYrK5XKpVCqfzz88fFcIdnvi'
    '+lpaX1+fnp5Gmng8zrKsx+OJxWKAwrxeL+4LhYIkiYqkycm3PI8iMgZaLBYh'
    'BiCQS6VSIpFAbohcXV2F//z8QFHvFMLJyeH4+DhiyWSS53nEwISk4+PjUCiU'
    'yWS2tray2awgFBYX88TvFy2WUYwPLQqCgE7S6bSaHl+YOgCXyxUMvsFIidv9'
    'AXoA1Wq1Op1Oo9EwDDMyMtJoNPr6+lCh+xSBADM66icezye1adAMBsM/bz0w'
    'MOB2u+/v71dWVs7OPjudKQJZHBeDeo7jzs/P1RXoGuYLPbu7u6DhEnKIy5WF'
    'd3p6CtEQCmHLy8tGoxHVaJo2mUyYKWagpgD4z/JVKl8fH6t3d+WrK7Qbdzgo'
    '+JlMUpbbtdq3bkFl+bDeCwv7s7P78/Mfp6a4paVDCEUmhyOJg0uEZmbe4wAG'
    '8G8cz71zGGIwnwAAAABJRU5ErkJggqVIPg4=')    

ERROR_COLUMN = 0
EYE_COLUMN = 1
MODULE_NAME_COLUMN = 2
ERROR = "error"
OK = "ok"
EYE = "eye"
CLOSED_EYE = "closedeye"

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
        error_bitmap      = plv_get_bitmap(IMG_ERROR)
        ok_bitmap         = plv_get_bitmap(IMG_OK)
        eye_bitmap        = plv_get_bitmap(IMG_EYE)
        closed_eye_bitmap = plv_get_bitmap(IMG_CLOSED_EYE)
        error_dictionary = {ERROR:error_bitmap, OK:ok_bitmap}
        eye_dictionary   = {EYE:eye_bitmap, CLOSED_EYE:closed_eye_bitmap}
        cpgrid.hook_grid_button_column(grid, ERROR_COLUMN, 
                                       error_dictionary, hook_events=False)
        cpgrid.hook_grid_button_column(grid, EYE_COLUMN, eye_dictionary,
                                       hook_events=False)
        name_attrs = wx.grid.GridCellAttr()
        name_attrs.SetReadOnly(True)
        grid.SetColAttr(MODULE_NAME_COLUMN, name_attrs)
        wx.grid.EVT_GRID_CELL_LEFT_CLICK(grid, self.__on_grid_left_click)
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
        self.select_one_module(event.Row+1)
        if event.Col == EYE_COLUMN:
            if len(self.__pipeline.modules()) > event.Row:
                module = self.__pipeline.modules()[event.Row]
                module.show_frame = not module.show_frame
    
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
        if event.direction == cellprofiler.pipeline.DIRECTION_UP:
            old_index = event.module_num
        else:
            old_index = event.module_num - 2
        new_index = event.module_num - 1
        selected = self.__grid.IsInSelection(old_index, MODULE_NAME_COLUMN)
        self.__populate_row(pipeline.modules()[old_index])
        self.__populate_row(pipeline.modules()[new_index])
        if selected:
            self.__grid.SelectBlock(new_index, MODULE_NAME_COLUMN,
                                    new_index, MODULE_NAME_COLUMN,
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
                    if self.__pipeline_movie.slider.value >= module.module_num:
                        self.__pipeline_movie.slider.value = module.module_num - 1
                    return
                
    def on_stop_debugging(self):
        self.__pipeline_movie.slider.value = 0
        
    def on_idle(self,event):
        modules = self.__pipeline.modules()
        for idx,module in enumerate(modules):
            try:
                module.test_valid(self.__pipeline)
                target_name = module.module_name
                ec_value = OK
            except:
                ec_value = ERROR
            if ec_value != self.__grid.GetCellValue(idx,ERROR_COLUMN):
                self.__grid.SetCellValue(idx,ERROR_COLUMN,ec_value)
            if module.show_frame:
                eye_value = EYE
            else:
                eye_value = CLOSED_EYE
            if eye_value != self.__grid.GetCellValue(idx, EYE_COLUMN):
                self.__grid.SetCellValue(idx,EYE_COLUMN, eye_value)
