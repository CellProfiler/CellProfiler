import logging
import os
from urllib.parse import urlparse

import imageio
from cellprofiler_core.readers.imageio_reader import ImageIOReader
from google.cloud import storage

SUPPORTED_EXTENSIONS = {'.tiff'}


class GcsReader(ImageIOReader):
  """ Reads images from bucket(s) in Google Cloud Storage via user's Application Default Credential (ADC).
  Leverages ImageIOReader image processing methods.

  Prerequisites:
  User has authenticated with Google Cloud Storage Application Default Credential (ADC) by running command,
  `gcloud auth application-default login`,
  or is running CellProfiler in an environment where this credential has already been configured for them
  such as app.terra.bio or CloudShell.
  """

  reader_name = "GcsReader"

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
      ."""
      if image_file.url.lower().startswith("gs:") and image_file.file_extension in SUPPORTED_EXTENSIONS:
        return 1
      return -1

  def get_reader(self, volume=False):
    # Download image from Google Cloud Storage bucket.
    path_to_image = self.download_blob(self.file._url)
    if self._reader is None or volume != self._volume:
        if volume:
            self._reader = imageio.get_reader(path_to_image, mode='v')
        else:
            self._reader = imageio.get_reader(path_to_image, mode='i')
        self._volume = volume
    return self._reader

  @classmethod
  def supports_url(cls):
    return True

  def decode_gcs_url(self, url):
    p = urlparse(url)
    path = p.path.split('/', 1)
    bucket, file_path = p.netloc, path[1:][0]
    return bucket, file_path

  def download_blob(self, url):
    if url:
      # Create client to access Google Cloud Storage.
      client = storage.Client()
      # Get bucket name, file path from URL.
      bucket_id, file_path = self.decode_gcs_url(url)
      # Get bucket.
      bucket = client.bucket(bucket_id)
      # Download blob object to local file path.
      blob = bucket.blob(file_path)
      local_file_path = os.path.basename(file_path)
      blob.download_to_filename(local_file_path)
      return local_file_path