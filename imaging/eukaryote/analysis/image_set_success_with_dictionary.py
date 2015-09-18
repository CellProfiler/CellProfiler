from __future__ import with_statement
import imaging.eukaryote.analysis.image_set_success


class ImageSetSuccessWithDictionary(imaging.eukaryote.analysis.image_set_success.ImageSetSuccess):
    def __init__(self, analysis_id, image_set_number, shared_dicts):
        imaging.eukaryote.analysis.image_set_success.ImageSetSuccess.__init__(self, analysis_id, image_set_number=image_set_number)

        self.shared_dicts = shared_dicts
