from ._image_set_success import ImageSetSuccess


class ImageSetSuccessWithDictionary(ImageSetSuccess):
    def __init__(self, analysis_id, image_set_number, shared_dicts):
        ImageSetSuccess.__init__(self, analysis_id, image_set_number=image_set_number)

        self.shared_dicts = shared_dicts
