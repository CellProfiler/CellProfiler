"""
Use this container to store extra entries which should be added to 'Plugins' menu by
CellProfiler plugin modules.

Each entry should be a tuple containing (callback_fn, wx_id, name, tooltip).
callback_fn - A callback function to be triggered when the menu item is activated.
              Will receive a wx.Event object as an argument.
wx_id - a unique wx.NewId() which will be bound to a GUI function related to your plugin.
name - The label associated with the option in the menu.
tooltip - Expanded help message (not displayed on all platforms).

"""

PLUGIN_MENU_ENTRIES = []
