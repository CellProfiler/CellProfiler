class AbstractPipelineEvent:
    """Something that happened to the pipeline and was indicated to the listeners
    """

    def __init__(self, is_pipeline_modification=False, is_image_set_modification=False):
        self.is_pipeline_modification = is_pipeline_modification
        self.is_image_set_modification = is_image_set_modification

    def event_type(self):
        raise NotImplementedError(
            "AbstractPipelineEvent does not implement an event type"
        )


class PipelineLoadedEvent(AbstractPipelineEvent):
    """Indicates that the pipeline has been (re)loaded

    """

    def __init__(self):
        super(PipelineLoadedEvent, self).__init__(
            is_pipeline_modification=True, is_image_set_modification=True
        )

    def event_type(self):
        return "PipelineLoaded"


class PipelineClearedEvent(AbstractPipelineEvent):
    """Indicates that all modules have been removed from the pipeline

    """

    def __init__(self):
        super(PipelineClearedEvent, self).__init__(
            is_pipeline_modification=True, is_image_set_modification=True
        )

    def event_type(self):
        return "PipelineCleared"


class ModuleMovedPipelineEvent(AbstractPipelineEvent):
    """A module moved up or down

    """

    def __init__(self, module_num, direction, is_image_set_modification):
        super(ModuleMovedPipelineEvent, self).__init__(
            is_pipeline_modification=True,
            is_image_set_modification=is_image_set_modification,
        )
        self.module_num = module_num
        self.direction = direction

    def event_type(self):
        return "Module moved"


class ModuleAddedPipelineEvent(AbstractPipelineEvent):
    """A module was added to the pipeline

    """

    def __init__(self, module_num, is_image_set_modification=False):
        super(ModuleAddedPipelineEvent, self).__init__(
            is_pipeline_modification=True,
            is_image_set_modification=is_image_set_modification,
        )
        self.module_num = module_num

    def event_type(self):
        return "Module Added"


class ModuleRemovedPipelineEvent(AbstractPipelineEvent):
    """A module was removed from the pipeline

    """

    def __init__(self, module_num, is_image_set_modification=False):
        super(ModuleRemovedPipelineEvent, self).__init__(
            is_pipeline_modification=True,
            is_image_set_modification=is_image_set_modification,
        )
        self.module_num = module_num

    def event_type(self):
        return "Module deleted"


class ModuleEditedPipelineEvent(AbstractPipelineEvent):
    """A module had its settings changed

    """

    def __init__(self, module_num, is_image_set_modification=False):
        super(ModuleEditedPipelineEvent, self).__init__(
            is_pipeline_modification=True,
            is_image_set_modification=is_image_set_modification,
        )
        self.module_num = module_num

    def event_type(self):
        return "Module edited"


class URLsAddedEvent(AbstractPipelineEvent):
    def __init__(self, urls):
        super(self.__class__, self).__init__()
        self.urls = urls

    def event_type(self):
        return "URLs added to file list"


class URLsRemovedEvent(AbstractPipelineEvent):
    def __init__(self, urls):
        super(self.__class__, self).__init__()
        self.urls = urls

    def event_type(self):
        return "URLs removed from file list"


class FileWalkStartedEvent(AbstractPipelineEvent):
    def event_type(self):
        return "File walk started"


class FileWalkEndedEvent(AbstractPipelineEvent):
    def event_type(self):
        return "File walk ended"


class RunExceptionEvent(AbstractPipelineEvent):
    """An exception was caught during a pipeline run

    Initializer:
    error - exception that was thrown
    module - module that was executing
    tb - traceback at time of exception, e.g from sys.exc_info
    """

    def __init__(self, error, module, tb=None):
        self.error = error
        self.cancel_run = True
        self.skip_thisset = False
        self.module = module
        self.tb = tb

    def event_type(self):
        return "Pipeline run exception"


class PrepareRunExceptionEvent(RunExceptionEvent):
    """An event indicating an uncaught exception during the prepare_run phase"""

    def event_type(self):
        return "Prepare run exception"


class PostRunExceptionEvent(RunExceptionEvent):
    """An event indicating an uncaught exception during the post_run phase"""

    def event_type(self):
        return "Post run exception"


class PrepareRunErrorEvent(AbstractPipelineEvent):
    """A user configuration error prevented CP from running the pipeline

    Modules use this class to report conditions that prevent construction
    of the image set list. An example would be if the user misconfigured
    LoadImages or NamesAndTypes and no images were matched.
    """

    def __init__(self, module, message):
        super(self.__class__, self).__init__()
        self.module = module
        self.message = message


class LoadExceptionEvent(AbstractPipelineEvent):
    """An exception was caught during pipeline loading

    """

    def __init__(self, error, module, module_name=None, settings=None):
        self.error = error
        self.cancel_run = True
        self.module = module
        self.module_name = module_name
        self.settings = settings

    def event_type(self):
        return "Pipeline load exception"


class IPDLoadExceptionEvent(AbstractPipelineEvent):
    """An exception was cauaght while trying to load the image plane details

    This event is reported when an exception is thrown while loading
    the image plane details from the workspace's file list.
    """

    def __init__(self, error):
        super(self.__class__, self).__init__()
        self.error = error
        self.cancel_run = True

    def event_type(self):
        return "Image load exception"


class CancelledException(Exception):
    """Exception issued by the analysis worker indicating cancellation by UI

    This is here in order to solve some import dependency problems
    """

    pass


class PipelineLoadCancelledException(Exception):
    """Exception thrown if user cancels pipeline load"""

    pass


class EndRunEvent(AbstractPipelineEvent):
    """A run ended"""

    def event_type(self):
        return "Run ended"


class ModuleEnabledEvent(AbstractPipelineEvent):
    """A module was enabled

    module - the module that was enabled.
    """

    def __init__(self, module):
        """Constructor

        module - the module that was enabled
        """
        super(self.__class__, self).__init__(is_pipeline_modification=True)
        self.module = module

    def event_type(self):
        return "Module enabled"


class ModuleDisabledEvent(AbstractPipelineEvent):
    """A module was disabled

    module - the module that was disabled.
    """

    def __init__(self, module):
        """Constructor

        module - the module that was enabled
        """
        super(self.__class__, self).__init__(is_pipeline_modification=True)
        self.module = module

    def event_type(self):
        return "Module disabled"


class ModuleShowWindowEvent(AbstractPipelineEvent):
    """A module had its "show_window" state changed

    module - the module that had its state changed
    """

    def __init__(self, module):
        super(self.__class__, self).__init__(is_pipeline_modification=True)
        self.module = module

    def event_type(self):
        return "Module show_window changed"
