from ._pathname import Pathname


class URL(Pathname):
    """
    A setting that displays a path name or URL
    """

    def is_url(self):
        return any(
            [
                self.value_text.lower().startswith(scheme)
                for scheme in ("http:", "https:", "ftp:")
            ]
        )

    def test_valid(self, pipeline):
        if not self.is_url():
            super(URL, self).test_valid(pipeline)
