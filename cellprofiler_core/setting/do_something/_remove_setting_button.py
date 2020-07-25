from ._do_something import DoSomething


class RemoveSettingButton(DoSomething):
    """A button whose only purpose is to remove something from a list."""

    def __init__(self, text, label, list, entry, **kwargs):
        super(RemoveSettingButton, self).__init__(
            text, label, lambda: list.remove(entry), **kwargs
        )
