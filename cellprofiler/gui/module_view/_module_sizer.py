import logging

import wx

LOGGER = logging.getLogger(__name__)


class ModuleSizer(wx.Sizer):
    """The module sizer uses the maximum best width of the setting
    edit controls to compute the column widths, then it sets the text
    controls to wrap within the remaining space, then it uses the best
    height of each text control to lay out the rows.
    """

    def __init__(self, rows, cols=2):
        super(ModuleSizer, self).__init__()
        self.__rows = rows
        self.__cols = cols
        self.__min_text_width = 150
        self.__height_padding = 5
        self.__printed_exception = False
        self.__items = []

    def get_item(self, i, j):
        if len(self.__items) <= j or len(self.__items[j]) <= i:
            return None
        return self.__items[j][i]

    def Reset(self, rows, cols=3, destroy_windows=True):
        if destroy_windows:
            windows = []
            for j in range(self.__rows):
                for i in range(self.__cols):
                    item = self.get_item(i, j)
                    if item is None:
                        continue
                    if item.IsWindow():
                        window = item.GetWindow()
                        if isinstance(window, wx.Window):
                            windows.append(window)
            for window in windows:
                window.Hide()
                window.Destroy()
        self.Clear(False)
        self.__rows = rows
        self.__cols = cols
        self.__items = []

    def Add(self, control, *args, **kwargs):
        if len(self.__items) == 0 or len(self.__items[-1]) == self.__cols:
            self.__items.append([])
        item = super(ModuleSizer, self).Add(control, *args, **kwargs)
        self.__items[-1].append(item)
        return item

    def CalcMin(self):
        """Calculate the minimum from the edit controls.  Returns a
        wx.Size where the height is the total height of the grid and
        the width is self.__min_text_width plus the widths of the edit
        controls and help controls.
        """
        try:
            if (
                self.__rows * self.__cols == 0
                or self.Children is None
                or len(self.Children) == 0
            ):
                return wx.Size(0, 0)
            height = self.__height_padding
            for j in range(0, self.__rows):
                borders = [
                    self.get_item(col, j).GetBorder()
                    for col in range(2)
                    if self.get_item(col, j) is not None
                ]
                if len(borders) == 0:
                    height += 10
                else:
                    height_border = max(borders)
                    height += self.get_row_height(j) + 2 * height_border
            height += self.__height_padding
            self.__printed_exception = False
            return wx.Size(
                self.calc_edit_size()[0]
                + self.__min_text_width
                + self.calc_help_size()[0],
                height,
            )
        except:
            # This happens, hopefully transiently, on the Mac
            if not self.__printed_exception:
                LOGGER.error("WX internal error detected", exc_info=True)
                self.__printed_exception = True
            return wx.Size(0, 0)

    def get_row_height(self, j):
        height = 0
        for i in range(self.__cols):
            item = self.get_item(i, j)
            if item is None:
                continue
            if item.IsWindow() and isinstance(item.GetWindow(), wx.StaticLine):
                height = max(height, item.CalcMin()[1] * 1.25)
            else:
                height = max(height, item.CalcMin()[1])
        return height

    def calc_column_size(self, j):
        """Return a wx.Size with the total height of the controls in
        column j and the maximum of their widths.
        """
        height = 0
        width = 0
        for i in range(self.__rows):
            item = self.get_item(j, i)
            if item is None:
                continue
            size = item.CalcMin()
            height += size[1]
            width = max(width, size[0])
        return wx.Size(width, height)

    def calc_help_size(self):
        return self.calc_column_size(2)

    def calc_edit_size(self):
        return self.calc_column_size(1)

    def calc_max_text_width(self):
        width = self.__min_text_width
        for i in range(self.__rows):
            item = self.get_item(0, i)
            if item is None:
                continue
            control = item.GetWindow()
            assert isinstance(control, wx.StaticText), (
                "Control at column 0, "
                "%d of grid is not StaticText: %s" % (i, str(control))
            )
            text = control.GetLabel().replace("\n", " ")
            ctrl_width = control.GetFullTextExtent(text)[0] + 2 * item.GetBorder()
            width = max(width, ctrl_width)
        return width

    def RecalcSizes(self):
        """Recalculate the sizes of our items, resizing the text boxes
        as we go.
        """
        if self.__rows * self.__cols == 0:
            return
        try:
            size = self.GetSize()
            width = size[0] - 20
            edit_width = self.calc_edit_size()[0]
            help_width = self.calc_help_size()[0]
            max_text_width = self.calc_max_text_width()
            if edit_width + help_width + max_text_width < width:
                edit_width = width - max_text_width - help_width
            elif edit_width * 4 < width:
                edit_width = width / 4
            text_width = max([width - edit_width - help_width, self.__min_text_width])
            widths = [text_width, edit_width, help_width]
            #
            # Change all static text controls to wrap at the text width. Then
            # ask the items how high they are and do the layout of the line.
            #
            height = self.__height_padding
            panel = self.GetContainingWindow()
            for i in range(self.__rows):
                text_item = self.get_item(0, i)
                edit_item = self.get_item(1, i)
                if edit_item is None:
                    continue
                inner_text_width = text_width - 2 * text_item.GetBorder()
                control = text_item.GetWindow()
                assert isinstance(control, wx.StaticText), (
                    "Control at column 0, %d of grid is not StaticText: %s"
                    % (i, str(control))
                )
                text = control.GetLabel()
                edit_control = edit_item.GetWindow()
                height_border = max([x.GetBorder() for x in (edit_item, text_item)])
                if isinstance(edit_control, wx.StaticLine) and len(text) == 0:
                    #
                    # A line spans both columns
                    #
                    text_item.Show(False)
                    # make the divider height the same as a text row plus some
                    item_height = self.get_row_height(i)
                    assert isinstance(edit_item, wx.SizerItem)
                    border = edit_item.GetBorder()
                    third_width = (text_width + edit_width - 2 * border) / 3
                    item_location = wx.Point(
                        text_width - third_width / 2, height + border + item_height / 2
                    )
                    item_size = wx.Size(third_width, edit_item.GetSize()[1])
                    item_location = panel.CalcScrolledPosition(item_location)
                    edit_item.SetDimension(item_location, item_size)
                else:
                    text_item.Show(True)
                    if text_width > self.__min_text_width and (
                        text.find("\n") != -1
                        or control.GetFullTextExtent(text)[0] > inner_text_width
                    ):
                        text = text.replace("\n", " ")
                        control.SetLabel(text)
                        control.Wrap(inner_text_width)
                    row_height = self.get_row_height(i)
                    for j in range(self.__cols):
                        item = self.get_item(j, i)
                        item_x = sum(widths[0:j])
                        item_y = height
                        if (item.Flag & wx.EXPAND) == 0:
                            item_size = item.CalcMin()
                            if item.Flag & wx.ALIGN_CENTER_VERTICAL:
                                item_y = height + (row_height - item_size[1]) / 2
                            if item.Flag & wx.ALIGN_CENTER_HORIZONTAL:
                                item_x += (widths[j] - item_size[0]) / 2
                            elif item.Flag & wx.ALIGN_RIGHT:
                                item_x += widths[j] - item_size[0]
                        else:
                            item_size = wx.Size(widths[j], item.CalcMin()[1])
                        item_location = wx.Point(item_x, item_y)
                        item_location = panel.CalcScrolledPosition(item_location)
                        item.SetDimension(item_location, item_size)
                height += self.get_row_height(i) + 2 * height_border
        except:
            # This happens, hopefully transiently, on the Mac
            if not self.__printed_exception:
                LOGGER.warning("Detected WX error", exc_info=True)
                self.__printed_exception = True
