class DisplayPostRun:
    """Request a post-run display

    This is a message sent to the UI from the analysis worker"""

    def __init__(self, module_num, display_data):
        self.module_num = module_num
        self.display_data = display_data
