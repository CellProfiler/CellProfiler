"""CellProfilerGUI package

The CellProfilerGUI package holds the viewer and controller portions
of the cell profiler program

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

__version__="$Revision: 1 "

import base64
import cStringIO
import wx

def get_cp_image():
    """The CellProfiler icon as a wx.Image"""
    data64 = 'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAABIlJREFUSEutln9MW1UUxw/98e5re58UgTJkdIPRUhBY+x4/jIlmc5iQLYaoCFk0ahQWZKWwzZnFQdIGYRtDoIPS9lGgDBxrNDLjj7gYJULYlhlNwIjJEv8xGo0x8Q9N/EMintcHpXQdbSYvJ81tcu753O9555z7YC25Z25uLjnHWC9IcttOAmY/+fhu6k4CvrB3jTvPxzB2EvBZ94gbss7mVH5+ezGC2UnAbLd7CMwiY/WA8ZSxfGrYj5ib33yd5NuKcYvzkm+95bsEZi/lvVQYVe/3g6UZ2I6a+rHpyftgxAEs9U24weyjAhoykDROK9rBGIDCM5DaXnrw+gXvtZtfJgmLAzhf3eBTlsjRfZIOfpSWn2DNY7TcTwVRXYYJ9ED+WXiwTmNc7PJc6xkaDU3//PefEeTq6mpkHQdwAACPLEcfp8I7lL9K+Tc0JgQgLKJMpEKAtQ2BxQOFHigYBGM3GJxgqAGYm9/UFwdwCXLkzMzQ8lfZkn3qx3nyZA2TO8sJqACpUdmT3GSqV8cjshMeWP7hTnT2YgGDl8dGoAD3TFKhjjysZE6mkRY9satVzxlUR4I6kxilQyat83S8B/Km3N4EVXSrxz8MFgS8y9lAVZ9BmjPJsUzSrFc9pSeOdHX9e7TUHy6wLdEpL6rKuoqr737zsQrqSG6AlcT2a/MJ0xaOvg5ADEday9RlV7jNlyFj/DqhA/Tf/fFrAsC/a2tueAiziWXTrdlLGEcMAP+mMG1uTR6eAN0iNgpFg6c74xbuFgXTzr4xKAlylWgfcjYF02ogxw2kxUAceqbWQOyZ5HgqaatlD4RoueyGNq2rbIS0e7UFnNt4/MFxM0C9OudpRrLnmRyq2Msqq3SKCp2yilVYdIoqraJCq6wiKXuOMtmyG9ozoG945cX+gYFIKFy4XK6FhQWkbir49pcfA1AQpJWyzXBV9cTKkVNhBa16FSpwhNftoDz4ASfIblNaPH76Nl29Cbj4RINXWSLXdbi0+fc5q0JVm0Fa5CrC30zShFkCxSNXw40iGRTMjAUTAxZXlgYhO7rycB2ggktjBnVjJrHLgAzSlMrYq9nD09QmjRAN3whZ2w+ldQVvKvaJrDW6uuX+vEwFp6YIVEe16mfTiB2jg/oFlyY3EG43L5hC4kQCQDB05TAoR2D3MBSKGptPJ9XfxjCQGgrH0RQtFlSFpUz1Y6Smk80LchXSYNDyr22bfRm8rmDlt59Cnb1NKbtHIF8kVqnAo9oVx9wZ1hTiUJAV8yYdAn/BMuXsTTi0Yzv5o+ufns6tHAajyNrCs1pq1I1xjSN2XRw2mgPYhNG3lGm099z8fDtrwhSLujgApHqgOPjyyfsHyDsn+tyvAyeq90crkO+JEdhz4/vl/wvA/Xf++r3L+OgEFJ3Q4I0mpUi6qFnbMWpKJvo9UxSzufHQEQekR240D1gmO3p2EoCxhi+83Qe75JurH7L/STJ89CzafstXy0srN273QpbI8i+lFiQdP2rYbb9H/rLzuc5dAsNCd+y9mHgWJTxR5NPRuavionsgoX/E4T/EiGsa8AtrWQAAAABJRU5ErkJggg=='
    data = base64.b64decode(data64)
    stream = cStringIO.StringIO(data)
    image = wx.ImageFromStream(stream)
    return image

def get_cp_bitmap():
    """The CellProfiler icon as a wx.Bitmap"""
    bmp = wx.BitmapFromImage(get_cp_image())
    return bmp
    
def get_icon():
    """The CellProfiler icon as a wx.Icon"""
    icon = wx.EmptyIcon()
    icon.CopyFromBitmap(get_cp_bitmap())
    return icon

BV_DOWN = "down"
BV_UP   = "up"
def draw_bevel(dc, rect, width, state, shadow_pen = None, highlight_pen = None):
    """Draw a bevel within the rectangle so the inside looks raised or lowered
    
    dc - device context for drawing
    rect - draw the bevel within this rectangle
    width - the width of the bevel in pixels
    state - either BV_DOWN for a bevel with a lowered appearance or BV_UP
            for raised appearance
    shadow_pen - pen to use for drawing the shadow portion of the bevel
    highlight_pen - pen to use for drawing the light portion of the bevel
    
    returns the coordinates of the inside rectangle
    """
    if shadow_pen == None:
        shadow_pen = wx.Pen(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DSHADOW))
    if highlight_pen == None:
        highlight_pen = wx.Pen(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DHIGHLIGHT))
    top_left_pen = (state == BV_UP and highlight_pen) or shadow_pen
    bottom_right_pen = (state == BV_UP and shadow_pen) or highlight_pen
    for i in range(width):
        dc.Pen = top_left_pen
        dc.DrawLine(rect.Left,rect.Top,rect.Left,rect.Bottom)
        dc.DrawLine(rect.Left,rect.Top,rect.Right,rect.Top)
        dc.Pen = bottom_right_pen
        dc.DrawLine(rect.Right,rect.Bottom, rect.Left, rect.Bottom)
        dc.DrawLine(rect.Right,rect.Bottom, rect.Right, rect.Top)
        rect = wx.Rect(rect.Left+1, rect.Top+1, rect.width-2, rect.height-2)
    return rect
