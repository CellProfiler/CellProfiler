from ._abstract_image import AbstractImage


class CallbackImage(AbstractImage):
    """An image provider proxy that calls the indicated callback functions (presumably in your module) to implement the methods
    """

    def __init__(self, name, image_provider_fn):
        """Constructor
        name              - name returned by the Name method
        image_provider_fn - function called during ProvideImage with the arguments, image_set and the CallbackImageProvider instance
        """

        self.__name = name
        self.__image_provider_fn = image_provider_fn

    def provide_image(self, image_set):
        return self.__image_provider_fn(image_set, self)

    def get_name(self):
        return self.__name
