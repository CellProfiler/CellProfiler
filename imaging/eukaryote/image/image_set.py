class ImageSet(object):
    """Represents the images for a particular iteration of a pipeline
    
    An image set is composed of one image provider per image in the set.
    The image provider loads or creates an image, given a dictionary of keys
    (which might represent things like the plate/well for the image set or the
    frame number in a movie, etc.) 
    """

    def __init__(self, number, keys, legacy_fields):
        """Constructor: 
        number = image set index 
        keys = dictionary of key/value pairs that uniquely identify the image set
        """
        self.__image_providers = []
        self.__images = {}
        self.__keys = keys
        self.__number = number
        self.__legacy_fields = legacy_fields

    def get_number(self):
        """The (zero-based) image set index
        """
        return self.__number

    number = property(get_number)

    @property
    def image_number(self):
        '''The image number as used in measurements and the database'''
        return self.__number + 1

    def get_keys(self):
        """The keys that uniquely identify the image set
        """
        return self.__keys

    keys = property(get_keys)

    def get_image(self, name,
                  must_be_binary=False,
                  must_be_color=False,
                  must_be_grayscale=False,
                  must_be_rgb=False,
                  cache=True):
        """Return the image associated with the given name
        
        name - name of the image within the image_set
        must_be_color - raise an exception if not a color image
        must_be_grayscale - raise an exception if not a grayscale image
        must_be_rgb - raise an exception if 2-d or if # channels not 3 or 4,
                      discard alpha channel.
        """
        name = str(name)
        if not self.__images.has_key(name):
            image = self.get_image_provider(name).provide_image(self)
            if cache:
                self.__images[name] = image
        else:
            image = self.__images[name]
        if must_be_binary and image.pixel_data.ndim == 3:
            raise ValueError("Image must be binary, but it was color")
        if must_be_binary and image.pixel_data.dtype != np.bool:
            raise ValueError("Image was not binary")
        if must_be_color and image.pixel_data.ndim != 3:
            raise ValueError("Image must be color, but it was grayscale")
        if (must_be_grayscale and
                (image.pixel_data.ndim != 2)):
            pd = image.pixel_data
            if pd.shape[2] >= 3 and \
                    np.all(pd[:, :, 0] == pd[:, :, 1]) and \
                    np.all(pd[:, :, 0] == pd[:, :, 2]):
                return GrayscaleImage(image)
            raise ValueError("Image must be grayscale, but it was color")
        if must_be_grayscale and image.pixel_data.dtype.kind == 'b':
            return GrayscaleImage(image)
        if must_be_rgb:
            if image.pixel_data.ndim != 3:
                raise ValueError("Image must be RGB, but it was grayscale")
            elif image.pixel_data.shape[2] not in (3, 4):
                raise ValueError("Image must be RGB, but it had %d channels" %
                                 image.pixel_data.shape[2])
            elif image.pixel_data.shape[2] == 4:
                logger.warning("Discarding alpha channel.")
                return RGBImage(image)
        return image

    def get_providers(self):
        """The list of providers (populated during the image discovery phase)"""
        return self.__image_providers

    providers = property(get_providers)

    def get_image_provider(self, name):
        """Get a named image provider
        
        name - return the image provider with this name
        """
        providers = filter(lambda x: x.name == name, self.__image_providers)
        assert len(providers) > 0, "No provider of the %s image" % (name)
        assert len(providers) == 1, "More than one provider of the %s image" % (name)
        return providers[0]

    def remove_image_provider(self, name):
        """Remove a named image provider
        
        name - the name of the provider to remove
        """
        self.__image_providers = filter(lambda x: x.name != name,
                                        self.__image_providers)

    def clear_image(self, name):
        '''Remove the image memory associated with a provider
        
        name - the name of the provider
        '''
        self.get_image_provider(name).release_memory()
        if self.__images.has_key(name):
            del self.__images[name]

    def clear_cache(self):
        '''Remove all of the cached images'''
        self.__images.clear()

    def get_names(self):
        """Get the image provider names
        """
        return [provider.name for provider in self.providers]

    names = property(get_names)

    def get_legacy_fields(self):
        """Matlab modules can stick legacy junk into the Images handles field. Save it in this dictionary.
        
        """
        return self.__legacy_fields

    legacy_fields = property(get_legacy_fields)

    def add(self, name, image):
        old_providers = [provider for provider in self.providers
                         if provider.name == name]
        if len(old_providers) > 0:
            self.clear_image(name)
        for provider in old_providers:
            self.providers.remove(provider)
        provider = VanillaImageProvider(name, image)
        self.providers.append(provider)
