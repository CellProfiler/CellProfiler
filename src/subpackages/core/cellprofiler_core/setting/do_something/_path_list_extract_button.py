from ._do_something import DoSomething


class PathListExtractButton(DoSomething):
    """A setting that displays as a button which refreshes the path list"""

    def __init__(self, text, label, *args, **kwargs):
        DoSomething.__init__(self, text, label, self.fn_callback, *args, **kwargs)
        # callback set by module view
        self.callback = None

    def fn_callback(self, *args, **kwargs):
        if self.callback is not None:
            self.callback(*args, **kwargs)
