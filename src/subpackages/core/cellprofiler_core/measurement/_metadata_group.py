class MetadataGroup(dict):
    """A set of metadata tag values and the image set indexes that match

    The MetadataGroup object represents a group of image sets that
    have the same values for a given set of tags. For instance, if an
    experiment has metadata tags of "Plate", "Well" and "Site" and
    we form a metadata group of "Plate" and "Well", then each metadata
    group will have image set indexes of the images taken of a particular
    well
    """

    def __init__(self, tag_dictionary, image_numbers):
        super(MetadataGroup, self).__init__(tag_dictionary)
        self.__image_numbers = image_numbers

    @property
    def image_numbers(self):
        return self.__image_numbers

    def __setitem__(self, tag, value):
        raise NotImplementedError("The dictionary is read-only")
