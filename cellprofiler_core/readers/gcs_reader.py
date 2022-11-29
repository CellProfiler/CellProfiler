import io
from urllib.parse import urlparse
import logging

import imageio
from cellprofiler_core.readers.imageio_reader import ImageIOReader
from cellprofiler_core.readers.imageio_reader import SUPPORTED_EXTENSIONS as IMAGEIO_SUPPORTED_EXTENSIONS
from cellprofiler_core.readers.imageio_reader import SEMI_SUPPORTED_EXTENSIONS as IMAGEIO_SEMI_SUPPORTED_EXTENSIONS
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
from google.cloud.exceptions import NotFound

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.tiff'}
SEMI_SUPPORTED_EXTENSIONS = IMAGEIO_SUPPORTED_EXTENSIONS.union(IMAGEIO_SEMI_SUPPORTED_EXTENSIONS).difference(SUPPORTED_EXTENSIONS)
SUPPORTED_SCHEMES = {'gs'}


class GcsReader(ImageIOReader):
    """
    Reads images from bucket(s) in Google Cloud Storage via user's Application
    Default Credential (ADC).
    
    Once the file is downloaded from GCS, this reader leverages ImageIO Reader's
    image loading methods.

    Prerequisites:
    User has authenticated with Google Cloud Storage Application Default Credential
    (ADC) by running the command: "gcloud auth application-default login",
    or is running CellProfiler in an environment where this credential has already
    been configured for them such as app.terra.bio or CloudShell.
    """

    reader_name = "Google Cloud Storage"
    variable_revision_number = 1
    supported_filetypes = SUPPORTED_EXTENSIONS.union(SUPPORTED_EXTENSIONS)
    supported_schemes = SUPPORTED_SCHEMES

    @classmethod
    def supports_format(cls, image_file, allow_open=False, volume=False):
        """This function needs to evaluate whether a given ImageFile object
        can be read by this reader class.

        Return value should be an integer representing suitability:
        -1 - 'I can't read this at all'
        1 - 'I am the one true reader for this format, don't even bother checking any others'
        2 - 'I am well-suited to this format'
        3 - 'I can read this format, but I might not be the best',
        4 - 'I can give it a go, if you must'

        The allow_open parameter dictates whether the reader is permitted to read the file when
        making this decision. If False the decision should be made using file extension only.
        Any opened files should be closed before returning.

        The volume parameter specifies whether the reader will need to return a 3D array.
        """
        if image_file.scheme not in SUPPORTED_SCHEMES:
            return -1

        if image_file.file_extension in SUPPORTED_EXTENSIONS:
            return 1

        if image_file.file_extension in SEMI_SUPPORTED_EXTENSIONS:
            return 3

        return 4

    def get_reader(self, volume=False):
        image_resource = self.__download_blob(self.file.url)
        if self._reader is None or volume != self._volume:
            if volume:
                self._reader = imageio.get_reader(image_resource, mode='v')
            else:
                self._reader = imageio.get_reader(image_resource, mode='i')
            self._volume = volume
        return self._reader

    @classmethod
    def supports_url(cls):
        return True

    @staticmethod
    def get_settings():
        return []

    def __decode_gcs_url(self, url):
        p = urlparse(url)
        file_path = p.path[1:] if p.path.startswith("/") else p.path
        bucket_id = p.netloc
        return bucket_id, file_path
  
    def __download_blob(self, url):
        if url:
            # Create client to access Google Cloud Storage.
            try:
                client = storage.Client()
                # Get bucket name, file path from URL.
                bucket_id, file_path = self.__decode_gcs_url(url)
                # Get bucket.
                bucket = client.bucket(bucket_id)
                # Download blob object to local file path.
                blob = bucket.blob(file_path)
                blob_bytes = io.BytesIO(blob.download_as_bytes())
                return blob_bytes
            except DefaultCredentialsError as e:
               LOGGER.error(e.args[0])
               return url
            except NotFound as e:
                # 404 error
                LOGGER.error(e)
                return url
