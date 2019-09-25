from . import _setting


class Divider(_setting.Setting):
    """The divider setting inserts a vertical space, possibly with a horizontal line, in the GUI"""

    save_to_pipeline = False

    def __init__(self, text="", line=True, doc=None):
        super(Divider, self).__init__(text, "n/a", doc=doc)
        self.line = line and (text == "")
