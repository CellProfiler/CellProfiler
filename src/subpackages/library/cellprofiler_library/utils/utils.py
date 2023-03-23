# TODO: get rid of this, no longer needed
def convert_setting(gui_setting_str):
    """
    Convert GUI setting strings to something cellprofiler
    library compatible. That is, remove spaces and hyphens.
    """
    rep_list = ((" ", "_"), ("-", "_"))
    converted_str = gui_setting_str
    for replacement in rep_list:
        converted_str = converted_str.replace(*replacement)
    return converted_str
