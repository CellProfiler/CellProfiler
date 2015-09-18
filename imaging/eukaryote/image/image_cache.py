class ImageCache(object):
    '''An HDF5 cache that can store an image, mask or crop mask
    
    '''
    IC_MONOCHROME = "Monochrome"
    IC_COLOR = "Color"
    IC_5D = "5D"

    def __init__(self, image):
        '''Initialize with the image to control'''
        self.__backing_store = None
        self.__name = None
        if image.ndim == 2:
            self.__type = ImageCache.IC_MONOCHROME
            self.__image = image.reshape(1, 1, 1, image.shape[0], image.shape[1])
        elif image.ndim == 3:
            self.__type = ImageCache.IC_COLOR
            self.__image = image.transpose(2, 0, 1).reshape(
                image.shape[2], 1, 1, image.shape[0], image.shape[1])
        else:
            self.__type = ImageCache.IC_5D
            self.__image = image

    def is_cached(self):
        '''Return True if image is already cached by a backing store
        
        '''
        return self.__backing_store is not None

    def cache(self, name, backing_store):
        '''Cache an image into a backing store
        
        name - unique channel name of the image
        backing_store - an HDF5ImageSet
        '''
        self.__backing_store = backing_store
        self.__name = name
        self.__backing_store.set_image(self.__name, self.__image)
        del self.__image

    def get(self):
        '''Get the image in its original format'''
        if self.is_cached():
            image = self.__backing_store.get_image(self.__name)
        else:
            image = self.__image
        if self.__type == ImageCache.IC_MONOCHROME:
            return image.reshape(image.shape[3], image.shape[4])
        elif self.__type == ImageCache.IC_COLOR:
            return image.reshape(
                image.shape[0], image.shape[3], image.shape[4]).transpose(1, 2, 0)
