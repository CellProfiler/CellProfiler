class ValidationRequestController:
    """A request for module validation"""

    def __init__(self, pipeline, module, callback):
        """Initialize the validation request

        pipeline - pipeline in question
        module - module in question
        callback - call this callback if there is an error. Do it on the GUI thread
        """
        self.pipeline = pipeline
        self.module_num = module.module_num
        self.test_mode = pipeline.test_mode
        self.callback = callback
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
