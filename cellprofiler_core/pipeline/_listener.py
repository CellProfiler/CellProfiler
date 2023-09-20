class Listener:
    """A class to wrap add/remove listener for use with "with"

    Usage:
    def my_listener(pipeline, event):
        .....
    with pipeline.PipelineListener(pipeline, my_listener):
        # listener has been added
        .....
    # listener has been removed
    """

    def __init__(self, pipeline, listener):
        self.pipeline = pipeline
        self.listener = listener

    def __enter__(self):
        self.pipeline.add_listener(self.listener)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.pipeline.remove_listener(self.listener)
