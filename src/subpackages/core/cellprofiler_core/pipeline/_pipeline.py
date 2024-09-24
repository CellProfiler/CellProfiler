import bisect
import datetime
import gc
import hashlib
import io
import logging
import os
import os.path
import re
import sys
import tempfile
import timeit
import urllib.parse
import urllib.request
import requests

import uuid
import weakref
from ast import literal_eval
from packaging.version import Version

import numpy

from . import ImageFile
from .io import dump as dumpit
from ..constants.reader import ALL_READERS
from ..setting.multichoice import ImageNameSubscriberMultiChoice
from ..setting.subscriber import ImageSubscriber
from ..setting.subscriber import ImageListSubscriber
from ..utilities.core.modules import instantiate_module, reload_modules
from ..utilities.core.pipeline import read_file_list
from ._listener import Listener
from .dependency import ImageDependency
from .dependency import MeasurementDependency
from .dependency import ObjectDependency
from .event import CancelledException
from .event import EndRun
from .event import FileWalkEnded
from .event import IPDLoadException
from .event import LoadException
from .event import ModuleAdded
from .event import ModuleDisabled
from .event import ModuleEdited
from .event import ModuleEnabled
from .event import ModuleMoved
from .event import ModuleRemoved
from .event import ModuleShowWindow
from .event import PipelineCleared
from .event import PipelineLoaded
from .event import PostRunException
from .event import PrepareRunError
from .event import PrepareRunException
from .event import RunException
from .event import URLsAdded
from .event import URLsCleared
from .event import URLsRemoved
from ..constants.measurement import COLTYPE_FLOAT
from ..constants.measurement import COLTYPE_INTEGER
from ..constants.measurement import COLTYPE_LONGBLOB
from ..constants.measurement import COLTYPE_VARCHAR
from ..constants.measurement import C_FRAME
from ..constants.measurement import C_METADATA
from ..constants.measurement import C_SERIES
from ..constants.measurement import EXPERIMENT
from ..constants.measurement import IMAGE
from ..constants.measurement import IMAGE_NUMBER
from ..constants.measurement import MCA_AVAILABLE_POST_RUN
from ..constants.pipeline import DIRECTION_DOWN
from ..constants.pipeline import DIRECTION_UP
from ..constants.pipeline import EXIT_STATUS
from ..constants.pipeline import GROUP_INDEX
from ..constants.pipeline import GROUP_NUMBER
from ..constants.pipeline import GROUP_LENGTH
from ..constants.pipeline import H_DATE_REVISION
from ..constants.pipeline import H_GIT_HASH
from ..constants.pipeline import H_HAS_IMAGE_PLANE_DETAILS
from ..constants.pipeline import H_MESSAGE_FOR_USER
from ..constants.pipeline import H_MODULE_COUNT
from ..constants.pipeline import H_SVN_REVISION
from ..constants.pipeline import M_MODIFICATION_TIMESTAMP
from ..constants.pipeline import M_PIPELINE
from ..constants.pipeline import M_TIMESTAMP
from ..constants.pipeline import M_USER_PIPELINE
from ..constants.pipeline import M_VERSION
from ..constants.pipeline import NATIVE_VERSION
from ..constants.pipeline import SAD_PROOFPOINT_COOKIE
from ..constants.workspace import DISPOSITION_CANCEL
from ..constants.workspace import DISPOSITION_PAUSE
from ..constants.workspace import DISPOSITION_SKIP
from ..constants.image import PASSTHROUGH_SCHEMES
from ..constants.modules.metadata import X_AUTOMATIC_EXTRACTION
from ..image import ImageSetList
from ..measurement import Measurements
from ..object import ObjectSet
from ..preferences import get_always_continue, get_headless
from ..preferences import get_conserve_memory
from ..preferences import report_progress
from ..setting import Measurement
from ..setting.text import Name
from ..utilities.measurement import load_measurements
from ..workspace import Workspace

from cellprofiler_core import __version__ as core_version


LOGGER = logging.getLogger(__name__)


def _is_fp(x):
    return hasattr(x, "seek") and hasattr(x, "read")


class Pipeline:
    """A pipeline represents the modules that a user has put together
    to analyze their images.

    """

    def __init__(self):
        self.__filtered_image_plane_details_metadata_settings = None
        self.caption_for_user = None
        self.__modules = []
        self.__listeners = []
        self.__measurement_columns = {}
        self.__measurement_column_hash = None
        self.test_mode = False
        self.message_for_user = None
        self.__settings = []
        self.__undo_stack = []
        self.__undo_start = None
        # Unlike workspace.file_list, pipeline.file_list is a list of ImageFile objects
        # instead of raw URLs.
        self.__file_list = []
        #
        # A cookie that's shared between the workspace and pipeline
        # and is used to figure out whether the two are synchronized
        #
        self.__file_list_generation = None
        #
        # A cookie to let the metadata module know that the file list
        # has been edited (so extracted metadata is invalid).
        #
        self.file_list_edited = False
        #
        # The filtered file list is the list of URLS after filtering using
        # the Images module. The images settings are used to determine
        # whether the cache is valid
        #
        self.__filtered_file_list = []
        self.__filtered_file_list_images_settings = tuple()
        #
        # The image plane details are generated by the metadata module
        # from the file list
        #
        self.__image_plane_details = []
        self.__image_plane_details_metadata_settings = tuple()
        #
        # The image plane list represents the individual planes within
        # each image. It is generated by the Images module.
        # Each ImagePlane is linked to it's source ImageFile.
        #
        self.__image_plane_list = []

        self.__needs_headless_extraction = False
        
        self.__undo_stack = []
        self.__volumetric = False

        #
        # Dictionary mapping module objects to a list of image names
        # which are no longer required going forward in the pipeline.
        # Allows CellProfiler to 'forget' intermediates which are no
        # longer needed when the 'conserve memory' setting is enabled.
        #
        self.redundancy_map = None

    def set_volumetric(self, value):
        self.__volumetric = value

    def volumetric(self):
        return self.__volumetric
    
    def set_needs_headless_extraction(self, value):
        self.__needs_headless_extraction = value

    def needs_headless_extraction(self):
        return self.__needs_headless_extraction

    def copy(self, save_image_plane_details=True, preserve_module_state=False):
        """Create a copy of the pipeline modules and settings"""
        fd = io.StringIO()
        self.dump(fd, save_image_plane_details=save_image_plane_details)
        pipeline = Pipeline()
        fd.seek(0)
        pipeline.load(fd)
        if preserve_module_state:
            for orig_module, copied_module in zip(self.modules(), pipeline.modules()):
                copied_module.shared_state = orig_module.shared_state.copy()
        return pipeline

    def settings_hash(self, until_module=None, as_string=False):
        """Return a hash of the module settings

        This function can be used to invalidate a cached calculation
        that's based on pipeline settings - if the settings change, the
        hash changes and the calculation must be performed again.

        We use secure hashing functions which are really good at avoiding
        collisions for small changes in data.
        """
        h = hashlib.md5()
        for module in self.modules():
            h.update(module.module_name.encode("utf-8"))
            for setting in module.settings():
                h.update(setting.unicode_value.encode("utf-8"))
            if module.module_name == until_module:
                break
        if as_string:
            return h.hexdigest()
        return h.digest()

    @staticmethod
    def instantiate_module(module_name):
        # Needed to populate modules list in workers
        import cellprofiler_core.modules

        return instantiate_module(module_name)

    def reload_modules(self):
        """Reload modules from source, and attempt to update pipeline to new versions.
        Returns True if pipeline was successfully updated.

        """
        # clear previously seen errors on reload
        import cellprofiler_core.modules

        reload_modules()
        # attempt to reinstantiate pipeline with new modules
        try:
            self.copy()  # if this fails, we probably can't reload
            fd = io.StringIO()
            self.dump(fd)
            fd.seek(0)
            self.loadtxt(fd, raise_on_error=True)
            return True
        except Exception as e:
            LOGGER.warning(
                "Modules reloaded, but could not reinstantiate pipeline with new versions.",
                exc_info=True,
            )
            return False

    @staticmethod
    def is_pipeline_txt_file(filename):
        """Test a file to see if it can be loaded by Pipeline.loadtxt

        filename - path to the file

        returns True if the file starts with the CellProfiler cookie.
        """
        with open(filename, "r", encoding="utf-8") as fd:
            return Pipeline.is_pipeline_txt_fd(fd)

    @staticmethod
    def is_pipeline_txt_fd(fd):
        header = fd.read(1024)
        fd.seek(0)
        if header.startswith("CellProfiler Pipeline: http://www.cellprofiler.org"):
            return True
        if re.search(SAD_PROOFPOINT_COOKIE, header):
            LOGGER.info('print_emoji(":cat_crying_because_of_proofpoint:")')
            return True
        return False

    def load(self, fd_or_filename):
        """Load the pipeline from a file

        fd_or_filename - either the name of a file or a file-like object
        """
        self.__modules = []
        self.__undo_stack = []
        self.__undo_start = None

        filename = None

        if hasattr(fd_or_filename, "seek") and hasattr(fd_or_filename, "read"):
            fd = fd_or_filename
            needs_close = False
        elif hasattr(fd_or_filename, "read") and hasattr(fd_or_filename, "url"):
            # This is a URL file descriptor. Read into a StringIO so that
            # seek is available.
            fd = io.StringIO()
            while True:
                text = fd_or_filename.read()
                if len(text) == 0:
                    break
                fd.write(text)
            fd.seek(0)
            needs_close = False
        elif os.path.exists(fd_or_filename):
            fd = open(fd_or_filename, "r", encoding="utf-8")
            # Verify that we can read from the file in utf-8 mode
            try:
                fd.read()
                fd.seek(0)
            except UnicodeDecodeError:
                # Newer pipelines may need unicode encoding
                fd.close()
                fd = open(fd_or_filename, "r", encoding="unicode_escape")
            needs_close = True
            filename = fd_or_filename
        else:
            # Assume is string URL
            parsed_path = urllib.parse.urlparse(fd_or_filename)
            if len(parsed_path.scheme) < 2:
                raise IOError("Could not find file, " + fd_or_filename)
            fd = urllib.request.urlopen(fd_or_filename)
            return self.load(fd)

        if hasattr(fd, 'name') and fd.name.endswith('.json'):
            # Load a JSON format pipeline
            from cellprofiler_core.pipeline.io._v6 import load
            load(self, fd)
            # Perform proper pipeline setup after loading.
            self.__settings = [
                self.capture_module_settings(module) for module in
                self.modules(False)
            ]
            for module in self.modules(False):
                module.post_pipeline_load(self)
            self.notify_listeners(PipelineLoaded())
            self.__undo_stack = []
            return
        if Pipeline.is_pipeline_txt_fd(fd):
            self.loadtxt(fd)
            return

        if needs_close:
            fd.close()
        else:
            fd.seek(0)

        if filename is None:
            fid, filename = tempfile.mkstemp(".h5")
            fd_out = os.fdopen(fid, "wb")
            fd_out.write(fd.read())
            fd_out.close()
            self.load(filename)
            os.unlink(filename)
            return
        else:
            m = load_measurements(filename)
            pipeline_text = m.get_experiment_measurement(M_PIPELINE)
            pipeline_text = pipeline_text
            self.load(io.StringIO(pipeline_text))
            return

    @staticmethod
    def respond_to_version_mismatch_error(message):
        LOGGER.warning(message)

    def loadtxt(self, fp_or_filename, raise_on_error=False):
        """Load a pipeline from a text file

        fd_or_filename - either a path to a file or a file-descriptor-like
                         object.
        raise_on_error - if there is an error loading the pipeline, raise an
                         exception rather than generating a LoadException event.

        See savetxt for more comprehensive documentation.
        """
        self.__modules = []
        self.message_for_user = None

        if _is_fp(fp_or_filename):
            return self.__loads(fp_or_filename, raise_on_error)
        else:
            return self.__load(fp_or_filename, raise_on_error)

    def __load(self, fp, raise_on_error=False):
        with open(fp, "r", encoding="utf-8") as fd:
            return self.parse_pipeline(fd, raise_on_error)

    def __loads(self, s, raise_on_error=False):
        return self.parse_pipeline(s, raise_on_error)

    def parse_pipeline(self, fd, raise_on_error, count=sys.maxsize):
        header = readline(fd)
        if not self.is_pipeline_txt_fd(io.StringIO(header)):
            raise NotImplementedError('Invalid header: "%s"' % header)
        checksum, details, count, version = self.validate_pipeline_file(fd, count)
        new_modules = self.setup_modules(fd, count, raise_on_error)
        if details:
            self.clear_urls(add_undo=False)
            self.__file_list = [ImageFile(url) for url in read_file_list(fd)]
            self.__filtered_file_list_images_settings = None
        self.__modules = new_modules
        self.__settings = [
            self.capture_module_settings(module) for module in self.modules(False)
        ]
        for module in self.modules(False):
            module.post_pipeline_load(self)
        self.notify_listeners(PipelineLoaded())
        if details:
            self.notify_listeners(URLsAdded(self.__file_list))
        self.__undo_stack = []
        return checksum, version

    def validate_pipeline_file(self, fd, module_count):
        version = NATIVE_VERSION
        has_image_plane_details = False
        git_hash = None
        ver = Version(core_version)
        pipeline_version = ver.base_version
        current_version = int(f"{ver.major}{ver.minor}{ver.micro}")
        while True:
            line = readline(fd)

            if line is None:
                if module_count == 0:
                    break
                raise ValueError(
                    "Pipeline file unexpectedly truncated before module section"
                )
            elif len(line.strip()) == 0:
                break

            kwd, value = line.split(":")

            if kwd == "Version":
                version = int(value)
                if version > NATIVE_VERSION:
                    raise ValueError(
                        "Pipeline file version is {}.\nCellProfiler can only read version {} or less.\nPlease upgrade to the latest version of CellProfiler.".format(
                            version, NATIVE_VERSION
                        )
                    )
            elif kwd in (H_SVN_REVISION, H_DATE_REVISION,):
                pipeline_version = int(value)
            elif kwd == H_MODULE_COUNT:
                module_count = int(value)
            elif kwd == H_HAS_IMAGE_PLANE_DETAILS:
                has_image_plane_details = value == "True"
            elif kwd == H_MESSAGE_FOR_USER:
                value = value
                self.caption_for_user, self.message_for_user = value.split("|", 1)
            elif kwd == H_GIT_HASH:
                git_hash = value
            else:
                print(line)

        pipeline_version = self.validate_pipeline_version(
            current_version, git_hash, pipeline_version
        )

        return git_hash, has_image_plane_details, module_count, pipeline_version

    def validate_pipeline_version(self, current_version, git_hash, pipeline_version):
        # if "pipeline_version" is an actual dave revision, ie unicode time
        # at some point this changed to CellProfiler version (e.g. CP 4.2.4 would be 400)
        if 20080101000000 < pipeline_version < 30080101000000:
            # being optomistic... a millenium should be OK, no?
            second, minute, hour, day, month = [
                int(pipeline_version / (100 ** i)) % 100 for i in range(5)
            ]
            year = int(pipeline_version / (100 ** 5))
            pipeline_date = datetime.datetime(
                year, month, day, hour, minute, second
            ).strftime(" @ %c")
        else:
            pipeline_date = ""
        if pipeline_version > current_version:
            message = "Your pipeline version is {} but you are running CellProfiler version {}. Loading this pipeline may fail or have unpredictable results.".format(
                pipeline_version, current_version
            )

            self.respond_to_version_mismatch_error(message)
        else:
            if (not get_headless()) and pipeline_version < current_version:
                if git_hash is not None:
                    message = (
                        "\n\tYour pipeline was saved using an old version\n"
                        "\tof CellProfiler (rev {}{}).\n"
                        "\tThe current version of CellProfiler can load\n"
                        "\tand run this pipeline, but if you make changes\n"
                        "\tto it and save, the older version of CellProfiler\n"
                        "\t(perhaps the version your collaborator has?) may\n"
                        "\tnot be able to load it.\n\n"
                        "\tYou can ignore this warning if you do not plan to save\n"
                        "\tthis pipeline or if you will only use it with this or\n"
                        "\tlater versions of CellProfiler."
                    ).format(git_hash, pipeline_date)
                    LOGGER.warning(message)
                else:
                    # pipeline versions pre-3.0.0 have unpredictable formatting
                    if pipeline_version == 300:
                        pipeline_version = ".".join(
                            [version for version in str(pipeline_version)]
                        )
                    else:
                        pipeline_version = str(pipeline_version)

                    message = (
                        "Your pipeline was saved using an old version\n"
                        "of CellProfiler (version {}). The current version\n"
                        "of CellProfiler can load and run this pipeline, but\n"
                        "if you make changes to it and save, the older version\n"
                        "of CellProfiler (perhaps the version your collaborator\n"
                        "has?) may not be able to load it.\n\n"
                        "You can ignore this warning if you do not plan to save\n"
                        "this pipeline or if you will only use it with this or\n"
                        "later versions of CellProfiler."
                    ).format(pipeline_version)
                    LOGGER.warning(message)

        return pipeline_version

    def setup_modules(self, fd, module_count, raise_on_error):
        from cellprofiler_core.modules.metadata import Metadata
        from cellprofiler_core.modules.images import Images
        #
        # The module section
        #
        new_modules = []
        module_number = 1
        skip_attributes = ["svn_version", "module_num"]
        for i in range(module_count):
            line = readline(fd)
            if line is None:
                break
            settings = []
            try:
                module = None
                module_name = None
                split_loc = line.find(":")
                if split_loc == -1:
                    raise ValueError("Invalid format for module header: %s" % line)
                module_name = line[:split_loc].strip()
                attribute_string = line[(split_loc + 1) :]
                #
                # Decode the settings
                #
                last_module = False
                while True:
                    line = readline(fd)
                    if line is None:
                        last_module = True
                        break
                    if len(line.strip()) == 0:
                        break
                    if len(line.split(":", 1)) != 2:
                        raise ValueError("Invalid format for setting: %s" % line)
                    text, setting = line.split(":", 1)
                    # handle metadata from v3 pipelines
                    if r"\\\\g<" in setting:
                        if r"\x7C" in setting:
                            setting = setting.replace(r"\x7C\\\\g<", r"|\g<")
                        else:
                            setting = setting.replace(r"\\\\g<", r"\g<")
                    if "\\x" in setting:
                        try:
                            setting = setting.encode("utf-8").decode("unicode_escape")
                        except UnicodeDecodeError as e:
                            setting = setting.encode("utf-8").decode("utf-8")
                    if len(setting) > 1:
                        setting = setting.replace("\x00", "").strip("ÿþ")
                    settings.append(setting)

                module = self.setup_module(
                    attribute_string,
                    module_name=module_name,
                    module_number=module_number,
                    settings=settings,
                    skip_attributes=skip_attributes,
                )
            except Exception as instance:
                if raise_on_error:
                    raise
                LOGGER.error("Failed to load pipeline", exc_info=True)
                event = LoadException(instance, module, module_name, settings)
                self.notify_listeners(event)
                if event.cancel_run:
                    break
            if module is not None:
                new_modules.append(module)
                module_number += 1
            if isinstance(module, Metadata) and module.removed_automatic_extraction:
                # turn on extraction in Images, if metadata module was enabled
                if isinstance(new_modules[0], Images) and module.wants_metadata.value:
                    new_modules[0].want_split.value = True
                # now disable metdata module if the only extraction group was
                # extracting from image file headers; and reset the default
                # extraction method
                if module.extraction_method_count.value == 0:
                    module.wants_metadata.value = 'No'
                    module.add_extraction_method(False)
        return new_modules

    def setup_module(
        self, attribute_string, module_name, module_number, settings, skip_attributes
    ):
        #
        # Set up the module
        #
        module = self.instantiate_module(module_name)
        module.set_module_num(module_number)
        #
        # Decode the attributes. These are turned into strings using
        # repr, so True -> 'True', etc.
        #
        if (
            len(attribute_string) < 2
            or attribute_string[0] != "["
            or attribute_string[-1] != "]"
        ):
            raise ValueError("Invalid format for attributes: %s" % attribute_string)
        # Fix for array dtypes which contain split separator
        attribute_string = attribute_string.replace("dtype='|S", "dtype='S")
        if len(re.split("(?P<note>\|notes:\[.*?\]\|)",attribute_string)) ==3:
            # 4674- sometimes notes have pipes
            prenote,note,postnote = re.split("(?P<note>\|notes:\[.*?\]\|)",attribute_string)
            attribute_strings = prenote[1:].split("|")
            attribute_strings += [note[1:-1]]
            attribute_strings += postnote[:-1].split("|") 
        else:
            #old or weird pipeline without notes
            attribute_strings = attribute_string[1:-1].split("|")
        variable_revision_number = None

        for a in attribute_strings:
            if a.isascii():
                a = a.encode("utf-8").decode("unicode_escape")
            else:
                # We're in unicode already, remove escapes
                a = a.replace("\\", "")
            if len(a.split(":", 1)) != 2:
                raise ValueError("Invalid attribute string: %s" % a)
            attribute, value = a.split(":", 1)
            if attribute == "notes":
                try:
                    value = literal_eval(value)
                except SyntaxError as e:
                    value = value[1:-1].replace('"', "").replace("'", "").split(",")
                if any([chr(226) in line for line in value]):
                    # There were some unusual UTF-8 characters present, let's try to fix them.
                    try:
                        value = [
                            line.encode("raw-unicode-escape").decode("utf-8")
                            for line in value
                        ]
                    except Exception as e:
                        LOGGER.error(
                            f"Error during notes decoding\n\t{e}\n\tSome characters may have been lost"
                        )
            elif attribute == "batch_state":
                value = numpy.zeros((0,), numpy.uint8)
            else:
                value = literal_eval(value)
            if attribute in skip_attributes:
                continue

            if attribute == "variable_revision_number":
                variable_revision_number = value
            else:
                setattr(module, attribute, value)
        if variable_revision_number is None:
            raise ValueError(
                "Module %s did not have a variable revision # attribute" % module_name
            )
        module.set_settings_from_values(settings, variable_revision_number, module_name)
        if module_name == "NamesAndTypes":
            self.__volumetric = module.process_as_3d.value

        return module

    def dump(self, fp, save_image_plane_details=True, sanitize=False):
        dumpit(self, fp, save_image_plane_details, sanitize=sanitize)

    def save_pipeline_notes(self, fd, indent=2):
        """Save pipeline notes to a text file

        fd - file descriptor of the file.

        indent - indent of the notes relative to module header.
        """
        lines = []
        for module in self.modules(exclude_disabled=False):
            if module.enabled:
                fmt = "[%4.d] [%s]"
            else:
                fmt = "[%4.d] [%s] (disabled)"
            lines.append(fmt % (module.module_num, module.module_name))
            for note in module.notes:
                lines.append("%s%s" % ("".join([" "] * indent), note))
            lines.append("")
        fd.write("\n".join(lines))

    def save_pipeline_citations(self, fd, indent=2):
        """Save pipeline citations to a text file

        fd - file descriptor of the file.

        indent - indent of the notes relative to module header.
        """
        lines = []
        headers = {
                      'Accept': 'text/x-bibliography; style=apa',
                    }
        
        lines.append("Please cite the following when using CellProfiler: Stirling, D. R., Swain-Bowden, M. J., Lucas, A. M., Carpenter, A. E., Cimini, B. A., & Goodman, A. (2021). CellProfiler 4: improvements in speed, utility and usability. In BMC Bioinformatics (Vol. 22, Issue 1). Springer Science and Business Media LLC. https://doi.org/10.1186/s12859-021-04344-9")
        lines.append("")
        for module in self.modules(exclude_disabled=False):
            if module.enabled:
                fmt = "[%4.d] [%s]"
            else:
                fmt = "[%4.d] [%s] (disabled)"
            doi_link_list=module.doi
            if len(doi_link_list)>0:
                citation_list = [
                    f"{doi_text} {requests.get(doi_link, headers=headers).content.decode('utf-8')}"
                    for doi_text, doi_link in doi_link_list.items()
                ]
                
                lines.append(fmt % (module.module_num, module.module_name))
                lines.append("\n".join(citation_list))
                
                lines.append("")
        fd.write("\n".join(lines))

    def write_pipeline_measurement(self, m, user_pipeline=False):
        """Write the pipeline experiment measurement to the measurements

        m - write into these measurements

        user_pipeline - if True, write the pipeline into M_USER_PIPELINE
                        M_USER_PIPELINE is the pipeline that should be loaded
                        by the UI for the user for cases like a pipeline
                        created by CreateBatchFiles.
        """
        assert isinstance(m, Measurements)
        fd = io.StringIO()
        self.dump(fd, save_image_plane_details=False)
        m.add_measurement(
            EXPERIMENT, M_USER_PIPELINE if user_pipeline else M_PIPELINE, fd.getvalue(),
        )

    def clear_measurements(self, m):
        """Erase all measurements, but make sure to re-establish the pipeline one

        m - measurements to be cleared
        """
        m.clear()
        self.write_experiment_measurements(m)

    def requires_aggregation(self):
        """Return True if the pipeline requires aggregation across image sets

        If a pipeline has aggregation modules, the image sets in a group
        need to be run sequentially on the same worker.
        """
        for module in self.modules():
            if module.is_aggregation_module():
                return True
        return False

    def obfuscate(self):
        """Tell all modules in the pipeline to obfuscate any sensitive info

        This call is designed to erase any information that users might
        not like to see uploaded. You should copy a pipeline before obfuscating.
        """
        for module in self.modules(False):
            module.obfuscate()

    def run(
        self,
        frame=None,
        image_set_start=1,
        image_set_end=None,
        grouping=None,
        measurements_filename=None,
        initial_measurements=None,
    ):
        """Run the pipeline

        Run the pipeline, returning the measurements made
        frame - the frame to be used when displaying graphics or None to
                run headless
        image_set_start - the image number of the first image to be run
        image_set_end - the index of the last image to be run + 1
        grouping - a dictionary that gives the keys and values in the
                   grouping to run or None to run all groupings
        measurements_filename - name of file to use for measurements
        """
        measurements = Measurements(
            image_set_start=image_set_start,
            filename=measurements_filename,
            copy=initial_measurements,
        )
        if not self.in_batch_mode() and initial_measurements is not None:
            #
            # Need file list in order to call prepare_run
            #
            from cellprofiler_core.utilities.hdf5_dict import HDF5FileList

            src = initial_measurements.hdf5_dict.hdf5_file
            dest = measurements.hdf5_dict.hdf5_file
            if HDF5FileList.has_file_list(src):
                HDF5FileList.copy(src, dest)
                self.add_urls(HDF5FileList(dest).get_filelist())

        measurements.is_first_image = True
        for m in self.run_with_yield(
            frame,
            image_set_start,
            image_set_end,
            grouping,
            run_in_background=False,
            initial_measurements=measurements,
        ):
            measurements = m
        return measurements

    def group(
        self, grouping, image_set_start, image_set_end, initial_measurements, workspace
    ):
        """Enumerate relevant image sets.  This function is side-effect free, so it can be called more than once."""

        keys, groupings = self.get_groupings(workspace)

        if grouping is not None and set(keys) != set(grouping.keys()):
            raise ValueError(
                "The grouping keys specified on the command line (%s) must be the same as those defined by the modules in the pipeline (%s)"
                % (", ".join(list(grouping.keys())), ", ".join(keys))
            )

        for gn, (grouping_keys, image_numbers) in enumerate(groupings):
            if grouping is not None and grouping != grouping_keys:
                continue

            need_to_run_prepare_group = True

            for gi, image_number in enumerate(image_numbers):
                if image_number < image_set_start:
                    continue

                if image_set_end is not None and image_number > image_set_end:
                    continue

                if initial_measurements is not None and all(
                    [
                        initial_measurements.has_feature(IMAGE, f)
                        for f in (GROUP_NUMBER, GROUP_INDEX,)
                    ]
                ):
                    group_number, group_index = [
                        initial_measurements[IMAGE, f, image_number,]
                        for f in (GROUP_NUMBER, GROUP_INDEX,)
                    ]
                else:
                    group_number = gn + 1

                    group_index = gi + 1

                if need_to_run_prepare_group:
                    yield group_number, group_index, image_number, lambda: self.prepare_group(
                        workspace, grouping_keys, image_numbers
                    )
                else:
                    yield group_number, group_index, image_number, lambda: True

                need_to_run_prepare_group = False

            if not need_to_run_prepare_group:
                yield None, None, None, lambda workspace: self.post_group(
                    workspace, grouping_keys
                )

    def run_with_yield(
        self,
        frame=None,
        image_set_start=1,
        image_set_end=None,
        grouping=None,
        run_in_background=True,
        status_callback=None,
        initial_measurements=None,
    ):
        """Run the pipeline, yielding periodically to keep the GUI alive.
        Yields the measurements made.

        Arguments:
           status_callback - None or a callable with arguments
                             (module, image_set) that will be called before
                             running each module.

        Run the pipeline, returning the measurements made
        """

        can_display = not get_headless()

        columns = self.get_measurement_columns()

        if image_set_start is not None:
            assert isinstance(
                image_set_start, int
            ), "Image set start must be an integer"

        if image_set_end is not None:
            assert isinstance(image_set_end, int), "Image set end must be an integer"

        if initial_measurements is None:
            measurements = Measurements(image_set_start)
        else:
            measurements = initial_measurements

        image_set_list = ImageSetList()

        workspace = Workspace(
            self, None, None, None, measurements, image_set_list, frame
        )
        
        if len(self.__modules)>1:
            from cellprofiler_core.modules.metadata import Metadata
            if type(self.__modules[1]) == Metadata:
                if self.__modules[1].wants_metadata.value:
                    for extraction_group in self.__modules[1].extraction_methods:
                        if extraction_group.extraction_method.value == X_AUTOMATIC_EXTRACTION:
                            self.set_needs_headless_extraction(True)
                            break

        try:
            if not self.prepare_run(workspace):
                return

            #
            # Remove image sets outside of the requested ranges
            #
            image_numbers = measurements.get_image_numbers()

            to_remove = []

            if image_set_start is not None:
                to_remove += [x for x in image_numbers if x < image_set_start]

                image_numbers = [x for x in image_numbers if x >= image_set_start]

            if image_set_end is not None:
                to_remove += [x for x in image_numbers if x > image_set_end]

                image_numbers = [x for x in image_numbers if x <= image_set_end]

            if grouping is not None:
                keys, groupings = self.get_groupings(workspace)

                for grouping_keys, grouping_image_numbers in groupings:
                    if grouping_keys != grouping:
                        to_remove += list(grouping_image_numbers)

            if len(to_remove) > 0 and measurements.has_feature("Image", IMAGE_NUMBER,):
                for image_number in numpy.unique(to_remove):
                    measurements.remove_measurement(
                        "Image", IMAGE_NUMBER, image_number,
                    )

            # Keep track of progress for the benefit of the progress window.
            num_image_sets = len(measurements.get_image_numbers())

            image_set_count = -1

            is_first_image_set = True

            last_image_number = None

            LOGGER.info("Times reported are CPU and Wall-clock times for each module")

            __group = self.group(
                grouping,
                image_set_start,
                image_set_end,
                initial_measurements,
                workspace,
            )

            for group_number, group_index, image_number, closure in __group:
                if image_number is None:
                    if not closure(workspace):
                        measurements.add_experiment_measurement(EXIT_STATUS, "Failure")

                        return

                    continue

                image_set_count += 1

                if not closure():
                    return

                last_image_number = image_number

                measurements.clear_cache()

                for provider in measurements.providers:
                    provider.release_memory()

                measurements.next_image_set(image_number)

                if is_first_image_set:
                    measurements.image_set_start = image_number

                    measurements.is_first_image = True

                    is_first_image_set = False

                measurements.group_number = group_number

                measurements.group_index = group_index

                numberof_windows = 0

                slot_number = 0

                object_set = ObjectSet()

                image_set = measurements

                outlines = {}

                should_write_measurements = True

                grids = None

                for module in self.modules():
                    if module.should_stop_writing_measurements():
                        should_write_measurements = False
                    else:
                        module_error_measurement = "ModuleError_%02d%s" % (
                            module.module_num,
                            module.module_name,
                        )

                        execution_time_measurement = "ExecutionTime_%02d%s" % (
                            module.module_num,
                            module.module_name,
                        )

                    failure = 1

                    exception = None

                    tb = None

                    frame_if_shown = frame if module.show_window else None

                    workspace = Workspace(
                        self,
                        module,
                        image_set,
                        object_set,
                        measurements,
                        image_set_list,
                        frame_if_shown,
                        outlines=outlines,
                    )

                    grids = workspace.set_grids(grids)

                    if status_callback:
                        status_callback(
                            module, len(self.modules()), image_set_count, num_image_sets
                        )

                    start_time = datetime.datetime.now()

                    os_times = os.times()
                    wall_t0 = timeit.default_timer()
                    cpu_t0 = sum(os_times[:-1])

                    try:
                        self.run_module(module, workspace)
                    except Exception as instance:
                        print("pipeline_exception")
                        LOGGER.error(
                            "Error detected during run of module %s",
                            module.module_name,
                            exc_info=True,
                        )

                        exception = instance

                        tb = sys.exc_info()[2]

                    yield measurements

                    os_times = os.times()
                    wall_t1 = timeit.default_timer()
                    cpu_t1 = sum(os_times[:-1])

                    cpu_delta_sec = max(0, cpu_t1 - cpu_t0)
                    wall_delta_sec = max(0, wall_t1 - wall_t0)

                    LOGGER.info(
                        "%s: Image # %d, module %s # %d: CPU_time = %.2f secs, Wall_time = %.2f secs"
                        % (
                            start_time.ctime(),
                            image_number,
                            module.module_name,
                            module.module_num,
                            cpu_delta_sec,
                            wall_delta_sec,
                        )
                    )

                    if module.show_window and can_display and (exception is None):
                        try:
                            fig = workspace.get_module_figure(module, image_number)

                            module.display(workspace, fig)

                            fig.Refresh()
                        except Exception as instance:
                            LOGGER.error(
                                "Failed to display results for module %s",
                                module.module_name,
                                exc_info=True,
                            )

                            exception = instance

                            tb = sys.exc_info()[2]

                    workspace.refresh()

                    failure = 0

                    if exception is not None:
                        if get_always_continue():
                            workspace.set_disposition(DISPOSITION_SKIP)

                            should_write_measurements = False

                        else:
                            event = RunException(exception, module, tb)

                            self.notify_listeners(event)

                            if event.cancel_run:
                                return
                            elif event.skip_thisset:
                                # Skip this image, continue to others
                                workspace.set_disposition(DISPOSITION_SKIP)

                                should_write_measurements = False

                                measurements = None

                    # Paradox: ExportToDatabase must write these columns in order
                    #  to complete, but in order to do so, the module needs to
                    #  have already completed. So we don't report them for it.
                    if module.module_name != "Restart" and should_write_measurements:
                        measurements.add_measurement(
                            "Image", module_error_measurement, numpy.array([failure])
                        )

                        measurements.add_measurement(
                            "Image",
                            execution_time_measurement,
                            numpy.array([float(cpu_delta_sec)]),
                        )

                    while (
                        workspace.disposition == DISPOSITION_PAUSE and frame is not None
                    ):
                        # try to leave measurements temporary file in a readable state
                        measurements.flush()

                        yield measurements

                    if workspace.disposition == DISPOSITION_SKIP:
                        break
                    elif workspace.disposition == DISPOSITION_CANCEL:
                        measurements.add_experiment_measurement(EXIT_STATUS, "Failure")

                        return
                if get_conserve_memory():
                    gc.collect()
            # Close cached readers.
            # This may play a big role with cluster deployments or long standing jobs
            # by freeing up memory and resources.
            for reader in ALL_READERS.values():
                reader.clear_cached_readers()
            if measurements is not None:
                workspace = Workspace(
                    self, None, None, None, measurements, image_set_list, frame
                )

                exit_status = self.post_run(workspace)

                #
                # Record the status after post_run
                #
                measurements.add_experiment_measurement(EXIT_STATUS, exit_status)
        finally:
            if measurements is not None:
                # XXX - We want to force the measurements to update the
                # underlying file, or else we get partially written HDF5
                # files.  There must be a better way to do this.
                measurements.flush()

                del measurements

            self.end_run()

    def run_image_set(
        self,
        measurements,
        image_set_number,
        interaction_handler,
        display_handler,
        cancel_handler,
    ):
        """Run the pipeline for a single image set storing the results in measurements.

        Arguments:
             measurements - source of image information, destination for results.
             image_set_number - what image to analyze.
             interaction_handler - callback (to be set in workspace) for
                 interaction requests
             display_handler - callback for display requests

             self.prepare_run() and self.prepare_group() must have already been called.

        Returns a workspace suitable for use in self.post_group()
        """
        measurements.next_image_set(image_set_number)
        measurements.group_number = measurements[
            "Image", GROUP_NUMBER,
        ]
        measurements.group_index = measurements[
            "Image", GROUP_INDEX,
        ]
        object_set = ObjectSet()
        image_set = measurements
        measurements.clear_cache()
        for provider in measurements.providers:
            provider.release_memory()
        outlines = {}
        grids = None
        should_write_measurements = True
        for module in self.modules():
            LOGGER.info(f"Running module {module.module_name} {module.module_num}")
            if module.should_stop_writing_measurements():
                should_write_measurements = False
            workspace = Workspace(
                self,
                module,
                image_set,
                object_set,
                measurements,
                None,
                outlines=outlines,
            )
            workspace.interaction_handler = interaction_handler
            workspace.cancel_handler = cancel_handler

            grids = workspace.set_grids(grids)

            start_time = datetime.datetime.now()
            os_times = os.times()
            wall_t0 = timeit.default_timer()
            cpu_t0 = sum(os_times[:-1])
            try:
                self.run_module(module, workspace)
                if module.show_window:
                    display_handler(module, workspace.display_data, image_set_number)
                try:
                    if self.redundancy_map is not None:
                        if module in self.redundancy_map and len(self.modules()) > module.module_num:
                            to_forget = self.redundancy_map[module]
                            for image_name in to_forget:
                                LOGGER.info(f"Releasing memory for redundant image {image_name}")
                                workspace.image_set.clear_image(image_name)
                        gc.collect()
                except Exception as e:
                    LOGGER.warning(f"Encountered error during memory cleanup: {e}")
            except CancelledException:
                # Analysis worker interaction handler is telling us that
                # the UI has cancelled the run. Forward exception upward.
                raise
            except Exception as exception:
                print("run_image_set_exception, get_always_continue",get_always_continue())
                LOGGER.error(
                    "Error detected during run of module %s#%d",
                    module.module_name,
                    module.module_num,
                    exc_info=True,
                )
                if should_write_measurements:
                    measurements[
                        "Image",
                        "ModuleError_%02d%s" % (module.module_num, module.module_name),
                    ] = 1
                if get_always_continue():
                    return
                evt = RunException(exception, module, sys.exc_info()[2])
                self.notify_listeners(evt)
                if evt.cancel_run or evt.skip_thisset:
                    # actual cancellation or skipping handled upstream.
                    return

            os_times = os.times()
            wall_t1 = timeit.default_timer()
            cpu_t1 = sum(os_times[:-1])
            cpu_delta_secs = max(0, cpu_t1 - cpu_t0)
            wall_delta_secs = max(0, wall_t1 - wall_t0)
            LOGGER.info(
                "%s: Image # %d, module %s # %d: CPU_time = %.2f secs, Wall_time = %.2f secs"
                % (
                    start_time.ctime(),
                    image_set_number,
                    module.module_name,
                    module.module_num,
                    cpu_delta_secs,
                    wall_delta_secs,
                )
            )
            # Paradox: ExportToDatabase must write these columns in order
            #  to complete, but in order to do so, the module needs to
            #  have already completed. So we don't report them for it.
            if should_write_measurements:
                measurements[
                    "Image",
                    "ModuleError_%02d%s" % (module.module_num, module.module_name),
                ] = 0
                measurements[
                    "Image",
                    "ExecutionTime_%02d%s" % (module.module_num, module.module_name),
                ] = float(cpu_delta_secs)

            measurements.flush()
            if workspace.disposition == DISPOSITION_SKIP:
                break

        if get_conserve_memory():
            gc.collect()

        return Workspace(
            self, None, measurements, object_set, measurements, None, outlines=outlines
        )

    def end_run(self):
        """Tell everyone that a run is ending"""
        self.notify_listeners(EndRun())

    def run_group_with_yield(
        self, workspace, grouping, image_numbers, stop_module, title, message
    ):
        """Run the modules for the image_numbers in a group up to an agg module

        This method runs a pipeline up to an aggregation step on behalf of
        an aggregation module. At present, you can call this within
        prepare_group to collect the images needed for aggregation.

        workspace - workspace containing the pipeline, image_set_list and
        measurements.

        grouping - the grouping dictionary passed into prepare_group.

        image_numbers - the image numbers that comprise the group

        stop_module - the aggregation module to stop at.

        The function yields the current workspace at the end of processing
        each image set. The workspace has a valid image_set and the
        measurements' image_number is the current image number.
        """
        m = workspace.measurements
        pipeline = workspace.pipeline
        image_set_list = workspace.image_set_list
        orig_image_number = m.image_set_number

        progress_dialog = self.create_progress_dialog(message, pipeline, title)
        if progress_dialog is not None:
            progress_dialog.SetRange(len(image_numbers))
        try:
            for i, image_number in enumerate(image_numbers):
                m.image_set_number = image_number
                image_set = m
                object_set = ObjectSet()
                old_providers = list(image_set.providers)
                for module in pipeline.modules():
                    w = Workspace(
                        self, module, image_set, object_set, m, image_set_list
                    )
                    if module == stop_module:
                        yield w
                        # Reset state of image set
                        del image_set.providers[:]
                        image_set.providers.extend(old_providers)
                        break
                    else:
                        self.run_module(module, w)
                    if progress_dialog is not None:
                        should_continue, skip = progress_dialog.Update(i + 1)
                        if not should_continue:
                            progress_dialog.EndModal(0)
                            return
        finally:
            if progress_dialog is not None:
                progress_dialog.Destroy()
            m.image_set_number = orig_image_number

    def calculate_last_image_uses(self):
        """
        Scans through the pipeline and produces a dict mapping each module to
        a list of images that can be safely forgotten after that module executes.
        Can be used to conserve system memory during a run - once an image is no
        longer required it no longer needs to be kept in memory
        """
        if not get_conserve_memory():
            self.redundancy_map = None
            return
        modules_to_names = weakref.WeakKeyDictionary()
        names_to_modules = {}
        for module in self.modules():
            for setting in module.settings():
                if isinstance(setting, ImageSubscriber):
                    if setting.value in ("None", None):
                        continue
                    names_to_modules[setting.value] = module
                elif isinstance(setting, ImageListSubscriber):
                    for item in setting.value:
                        if item in ("None", None):
                            continue
                        names_to_modules[item] = module
                elif isinstance(setting, ImageNameSubscriberMultiChoice):
                    for item in setting.get_selections():
                        if item in ("None", None):
                            continue
                        names_to_modules[item] = module
        for image_name, module_object in names_to_modules.items():
            if module_object in modules_to_names:
                modules_to_names[module_object].append(image_name)
            else:
                modules_to_names[module_object] = [image_name]
        self.redundancy_map = modules_to_names

    @staticmethod
    def create_progress_dialog(message, pipeline, title):
        return None

    @staticmethod
    def run_module(module, workspace):
        """Run one CellProfiler module

        Run the CellProfiler module with whatever preparation and cleanup
        needs to be done before and after.
        """
        module.run(workspace)

    def write_experiment_measurements(self, m):
        """Write the standard experiment measurments to the measurements file

        Write the pipeline, version # and timestamp.
        """
        assert isinstance(m, Measurements)
        self.write_pipeline_measurement(m)
        m.add_experiment_measurement(
            M_VERSION, core_version,
        )
        m.add_experiment_measurement(
            M_TIMESTAMP, datetime.datetime.now().isoformat(),
        )
        m.flush()

    def prepare_run(self, workspace, end_module=None):
        """Do "prepare_run" on each module to initialize the image_set_list

        workspace - workspace for the run

             pipeline - this pipeline

             image_set_list - the image set list for the run. The modules
                              should set this up with the image sets they need.
                              The caller can set test mode and
                              "combine_path_and_file" on the image set before
                              the call.

             measurements - the modules should record URL information here

             frame - the CPFigureFrame if not headless

             Returns True if all modules succeeded, False if any module reported
             failure or threw an exception

        test_mode - None = use pipeline's test mode, True or False to set explicitly

        end_module - if present, terminate before executing this module
        """
        assert isinstance(workspace, Workspace)
        m = workspace.measurements
        if self.has_legacy_loaders():
            # Legacy - there may be cached group number/group index
            #          image measurements which may be incorrect.
            m.remove_measurement(
                "Image", GROUP_INDEX,
            )
            m.remove_measurement(
                "Image", GROUP_NUMBER,
            )
        self.write_experiment_measurements(m)

        prepare_run_error_detected = False

        def on_pipeline_event(pipeline, event):
            nonlocal prepare_run_error_detected
            if isinstance(event, PrepareRunError):
                prepare_run_error_detected = True

        had_image_sets = False
        with Listener(self, on_pipeline_event):
            for module in self.modules():
                if module == end_module:
                    break
                try:
                    workspace.set_module(module)
                    if self.needs_headless_extraction():
                        from cellprofiler_core.modules.metadata import Metadata
                        if isinstance(module, Metadata):
                            workspace.file_list.add_files_to_filelist(self.file_list)
                            module.on_activated(workspace)
                            for extraction_group in module.extraction_methods:
                                if extraction_group.extraction_method.value == X_AUTOMATIC_EXTRACTION:
                                    module.do_update_metadata(extraction_group)
                    workspace.show_frame(module.show_window)
                    if (
                        not module.prepare_run(workspace)
                    ) or prepare_run_error_detected:
                        if workspace.measurements.image_set_count > 0:
                            had_image_sets = True
                        self.clear_measurements(workspace.measurements)
                        break
                except Exception as instance:
                    LOGGER.error(
                        "Failed to prepare run for module %s",
                        module.module_name,
                        exc_info=True,
                    )
                    event = PrepareRunException(instance, module, sys.exc_info()[2])
                    self.notify_listeners(event)
                    if event.cancel_run:
                        self.clear_measurements(workspace.measurements)
                        return False
        if workspace.measurements.image_set_count == 0:
            if not had_image_sets:
                self.report_prepare_run_error(
                    None,
                    "The pipeline did not identify any image sets.\n"
                    "Please correct any problems in your input module settings\n"
                    "and try again.",
                )
            return False

        if not m.has_feature("Image", GROUP_NUMBER,):
            # Legacy pipelines don't populate group # or index
            key_names, groupings = self.get_groupings(workspace)
            image_numbers = m.get_image_numbers()
            indexes = numpy.zeros(numpy.max(image_numbers) + 1, int)
            indexes[image_numbers] = numpy.arange(len(image_numbers))
            group_numbers = numpy.zeros(len(image_numbers), int)
            group_indexes = numpy.zeros(len(image_numbers), int)
            group_lengths = numpy.ones(len(image_numbers), int)
            for i, (key, group_image_numbers) in enumerate(groupings):
                iii = indexes[group_image_numbers]
                group_numbers[iii] = i + 1
                group_indexes[iii] = numpy.arange(len(iii)) + 1
                group_lengths[iii] = numpy.ones(len(iii), int) * len(iii)
            m.add_all_measurements(
                "Image", GROUP_NUMBER, group_numbers,
            )
            m.add_all_measurements(
                "Image", GROUP_INDEX, group_indexes,
            )
            m.add_all_measurements(
                "Image", GROUP_LENGTH, group_lengths,
            )
            #
            # The grouping for legacy pipelines may not be monotonically
            # increasing by group number and index.
            # We reorder here.
            #
            order = numpy.lexsort((group_indexes, group_numbers))
            if numpy.any(order[1:] != order[:-1] + 1):
                new_image_numbers = numpy.zeros(max(image_numbers) + 1, int)
                new_image_numbers[image_numbers[order]] = (
                    numpy.arange(len(image_numbers)) + 1
                )
                m.reorder_image_measurements(new_image_numbers)
        m.flush()

        if self.volumetric():
            unsupported = [
                module.module_name
                for module in self.__modules
                if not module.volumetric()
            ]

            if len(unsupported) > 0:
                self.report_prepare_run_error(
                    None,
                    "Cannot run pipeline. "
                    "The pipeline is configured to process data as 3D. "
                    "The pipeline contains modules which do not support 3D processing:"
                    "\n\n{}".format(", ".join(unsupported)),
                )

                return False

        return True

    def post_run(self, *args):
        """Do "post_run" on each module to perform aggregation tasks

        New interface:
        workspace - workspace with pipeline, module and measurements valid

        Old interface:

        measurements - the measurements for the run
        image_set_list - the image set list for the run
        frame - the topmost frame window or None if no GUI
        """
        from cellprofiler_core.module import Module

        if len(args) == 3:
            measurements, image_set_list, frame = args
            workspace = Workspace(
                self, None, None, None, measurements, image_set_list, frame
            )
        else:
            workspace = args[0]
        for module in self.modules():
            workspace.refresh()
            try:
                module.post_run(workspace)
            except Exception as instance:
                LOGGER.error(
                    "Failed to complete post_run processing for module %s."
                    % module.module_name,
                    exc_info=True,
                )
                event = PostRunException(instance, module, sys.exc_info()[2])
                self.notify_listeners(event)
                if event.cancel_run:
                    return "Failure"
            if (
                module.show_window
                and module.__class__.display_post_run != Module.display_post_run
            ):
                try:
                    workspace.post_run_display(module)
                except Exception as instance:
                    # Warn about display failure but keep going.
                    LOGGER.warning(
                        "Caught exception during post_run_display for module %s."
                        % module.module_name,
                        exc_info=True,
                    )
        workspace.measurements.add_experiment_measurement(
            M_MODIFICATION_TIMESTAMP, datetime.datetime.now().isoformat(),
        )

        return "Complete"

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        """Prepare to create a batch file

        This function is called when CellProfiler is about to create a
        file for batch processing. It will pickle the image set list's
        "legacy_fields" dictionary. This callback lets a module prepare for
        saving.

        workspace - the workspace to be saved
        fn_alter_path - this is a function that takes a pathname on the local
                        host and returns a pathname on the remote host. It
                        handles issues such as replacing backslashes and
                        mapping mountpoints. It should be called for every
                        pathname stored in the settings or legacy fields.
        """
        assert workspace.pipeline == self
        for module in self.modules():
            try:
                workspace.set_module(module)
                module.prepare_to_create_batch(workspace, fn_alter_path)
            except Exception as instance:
                LOGGER.error(
                    "Failed to collect batch information for module %s",
                    module.module_name,
                    exc_info=True,
                )
                event = RunException(instance, module, sys.exc_info()[2])
                self.notify_listeners(event)
                if event.cancel_run:
                    return

    def get_groupings(self, workspace):
        """Return the image groupings of the image sets in an image set list

        returns a tuple of key_names and group_list:
        key_names - the names of the keys that identify the groupings
        group_list - a sequence composed of two-tuples.
                     the first element of the tuple has the values for
                     the key_names for this group.
                     the second element of the tuple is a sequence of
                     image numbers comprising the image sets of the group
        For instance, an experiment might have key_names of 'Metadata_Row'
        and 'Metadata_Column' and a group_list of:
        [ (('A','01'), [0,96,192]),
          (('A','02'), [1,97,193]),... ]
        """
        groupings = None
        grouping_module = None
        for module in self.modules():
            workspace.set_module(module)
            new_groupings = module.get_groupings(workspace)
            if new_groupings is None:
                continue
            if groupings is None:
                groupings = new_groupings
                grouping_module = module
            else:
                raise ValueError(
                    "The pipeline has two grouping modules: # %d "
                    "(%s) and # %d (%s)"
                    % (
                        grouping_module.module_num,
                        grouping_module.module_name,
                        module.module_num,
                        module.module_name,
                    )
                )
        if groupings is None:
            return (), (((), workspace.measurements.get_image_numbers()),)
        return groupings

    def get_undefined_metadata_tags(self, pattern):
        """Find metadata tags not defined within the current measurements

        pattern - a regexp-like pattern that specifies how to insert
                  metadata into a string. Each token has the form:
                  "\(?<METADATA_TAG>\)" (matlab-style) or
                  "\g<METADATA_TAG>" (Python-style)
        """
        columns = self.get_measurement_columns()
        current_metadata = []
        for column in columns:
            object_name, feature, coltype = column[:3]
            if object_name == "Image" and feature.startswith(C_METADATA):
                current_metadata.append(feature[(len(C_METADATA) + 1) :])

        m = re.findall("\\(\\?[<](.+?)[>]\\)", pattern)
        if not m:
            m = re.findall("\\\\g[<](.+?)[>]", pattern)
        if m:
            m = list(
                filter(
                    (
                        lambda x: not any(
                            [x.startswith(y) for y in (C_SERIES, C_FRAME,)]
                        )
                    ),
                    m,
                )
            )
            undefined_tags = list(set(m).difference(current_metadata))
            return undefined_tags
        else:
            return []

    def prepare_group(self, workspace, grouping, image_numbers):
        """Prepare to start processing a new group

        workspace - the workspace containing the measurements and image set list
        grouping - a dictionary giving the keys and values for the group

        returns true if the group should be run
        """
        for module in self.modules():
            try:
                module.prepare_group(workspace, grouping, image_numbers)
            except Exception as instance:
                LOGGER.error(
                    "Failed to prepare group in module %s",
                    module.module_name,
                    exc_info=True,
                )
                event = RunException(instance, module, sys.exc_info()[2])
                self.notify_listeners(event)
                if event.cancel_run:
                    return False
        return True

    def post_group(self, workspace, grouping):
        """Do post-processing after a group completes

        workspace - the last workspace run
        """
        from cellprofiler_core.module import Module

        for module in self.modules():
            try:
                module.post_group(workspace, grouping)
            except Exception as instance:
                LOGGER.error(
                    "Failed during post-group processing for module %s"
                    % module.module_name,
                    exc_info=True,
                )
                event = RunException(instance, module, sys.exc_info()[2])
                self.notify_listeners(event)
                if event.cancel_run:
                    return False
            if (
                module.show_window
                and module.__class__.display_post_group != Module.display_post_group
            ):
                try:
                    workspace.post_group_display(module)
                except:
                    LOGGER.warning(
                        "Failed during post group display for module %s"
                        % module.module_name,
                        exc_info=True,
                    )
        return True

    def has_create_batch_module(self):
        for module in self.modules():
            if module.is_create_batch_module():
                return True
        return False

    def in_batch_mode(self):
        """Return True if the pipeline is in batch mode"""
        for module in self.modules():
            batch_mode = module.in_batch_mode()
            if batch_mode is not None:
                return batch_mode

    def turn_off_batch_mode(self):
        """Reset the pipeline to an editable state if batch mode is on

        A module is allowed to create hidden information that it uses
        to turn batch mode on or to save state to be used in batch mode.
        This call signals that the pipeline has been opened for editing,
        even if it is a batch pipeline; all modules should be restored
        to a state that's appropriate for creating a batch file, not
        for running a batch file.
        """
        for module in self.modules():
            module.turn_off_batch_mode()

    def clear(self):
        self.start_undoable_action()
        try:
            while len(self.__modules) > 0:
                self.remove_module(self.__modules[-1].module_num)
            self.notify_listeners(PipelineCleared())
            self.init_modules()
        finally:
            self.stop_undoable_action()

    def init_modules(self):
        """Initialize the module list

        Initialize the modules list to contain the four file modules.
        """
        from cellprofiler_core.modules.images import Images
        from cellprofiler_core.modules.metadata import Metadata
        from cellprofiler_core.modules.namesandtypes import NamesAndTypes
        from cellprofiler_core.modules.groups import Groups

        for i, module in enumerate((Images(), Metadata(), NamesAndTypes(), Groups())):
            module.set_module_num(i + 1)
            module.show_window = get_headless()
            self.add_module(module)

    def move_module(self, module_num, direction):
        """Move module # ModuleNum either DIRECTION_UP or DIRECTION_DOWN in the list

        Move the 1-indexed module either up one or down one in the list, displacing
        the other modules in the list
        """
        idx = module_num - 1
        if direction == DIRECTION_DOWN:
            if module_num >= len(self.__modules):
                raise ValueError(
                    "%(module_num)d is at or after the last module in the pipeline and can"
                    "t move down" % (locals())
                )
            module = self.__modules[idx]
            new_module_num = module_num + 1
            module.set_module_num(module_num + 1)
            next_module = self.__modules[idx + 1]
            next_module.set_module_num(module_num)
            self.__modules[idx] = next_module
            self.__modules[idx + 1] = module
            next_settings = self.__settings[idx + 1]
            self.__settings[idx + 1] = self.__settings[idx]
            self.__settings[idx] = next_settings
        elif direction == DIRECTION_UP:
            if module_num <= 1:
                raise ValueError(
                    "The module is at the top of the pipeline and can" "t move up"
                )
            module = self.__modules[idx]
            prev_module = self.__modules[idx - 1]
            new_module_num = prev_module.module_num
            module.module_num = new_module_num
            prev_module.module_num = module_num
            self.__modules[idx] = self.__modules[idx - 1]
            self.__modules[idx - 1] = module
            prev_settings = self.__settings[idx - 1]
            self.__settings[idx - 1] = self.__settings[idx]
            self.__settings[idx] = prev_settings
        else:
            raise ValueError("Unknown direction: %s" % direction)
        self.notify_listeners(ModuleMoved(new_module_num, direction, False))

        def undo():
            self.move_module(
                module.module_num,
                DIRECTION_DOWN if direction == DIRECTION_UP else DIRECTION_UP,
            )

        message = "Move %s %s" % (module.module_name, direction)
        self.__undo_stack.append((undo, message))

    def enable_module(self, module):
        """Enable a module = make it executable"""
        if module.enabled:
            LOGGER.warning(
                "Asked to enable module %s, but it was already enabled"
                % module.module_name
            )
            return
        module.enabled = True
        self.notify_listeners(ModuleEnabled(module))

        def undo():
            self.disable_module(module)

        message = "Enable %s" % module.module_name
        self.__undo_stack.append((undo, message))

    def disable_module(self, module):
        """Disable a module = prevent it from being executed"""
        if not module.enabled:
            LOGGER.warning(
                "Asked to disable module %s, but it was already disabled"
                % module.module_name
            )
        module.enabled = False
        self.notify_listeners(ModuleDisabled(module))

        def undo():
            self.enable_module(module)

        message = "Disable %s" % module.module_name
        self.__undo_stack.append((undo, message))

    def show_module_window(self, module, state=True):
        """Set the module's show_window state

        module - module to show or hide

        state - True to show, False to hide
        """
        if state != module.show_window:
            module.show_window = state
            self.notify_listeners(ModuleShowWindow(module))

            def undo():
                self.show_module_window(module, not state)

            message = "%s %s window" % (
                ("Show" if state else "Hide"),
                module.module_name,
            )
            self.__undo_stack.append((undo, message))

    def add_urls(self, urls, add_undo=True, metadata=None, series_names=None):
        """Add URLs to the file list

        urls - a collection of URLs
        add_undo - True to add the undo operation of this to the undo stack
        """
        real_list = []
        if metadata is not None:
            if series_names is None:
                # series_names should always be supplied with metadata, but just in case...
                series_names = [''] * len(urls)
            assert len(metadata) == len(series_names), "Metadata and series name arrays differ in length"
            urls, metadata, series_names = list(zip(*sorted(zip(urls, metadata, series_names), key=lambda x: x[0])))
        else:
            urls = sorted(urls)
        start = 0
        uid = uuid.uuid4()
        n = len(urls)
        for i, url in enumerate(urls):
            file_object = ImageFile(url)
            if metadata is not None:
                file_object.load_plane_metadata(metadata[i], names=series_names[i])
            if i % 100 == 0:
                path = urllib.parse.urlparse(url).path
                if "/" in path:
                    filename = path.rsplit("/", 1)[1]
                else:
                    filename = path
                filename = urllib.request.url2pathname(filename)
                report_progress(uid, float(i) / n, "Adding %s" % filename)
            pos = bisect.bisect_left(self.__file_list, url, start)
            if pos == len(self.file_list) or self.__file_list[pos] != url:
                real_list.append(url)
                self.__file_list.insert(pos, file_object)
            start = pos
        if n > 0:
            report_progress(uid, 1, "Done")
        # Invalidate caches
        self.__file_list_generation = uid
        self.__filtered_file_list_images_settings = None
        self.__image_plane_details_metadata_settings = None
        self.notify_listeners(URLsAdded(real_list))
        if add_undo:

            def undo():
                self.remove_urls(real_list)

            self.__undo_stack.append((undo, "Add images"))

    def remove_urls(self, urls):
        real_list = []
        urls = sorted(urls)
        start = 0
        for url in urls:
            pos = bisect.bisect_left(self.__file_list, url, start)
            if pos != len(self.__file_list) and self.__file_list[pos] == url:
                real_list.append(url)
                del self.__file_list[pos]
            start = pos
        if len(real_list):
            self.__filtered_file_list_images_settings = None
            self.__image_plane_details_metadata_settings = None
            self.__image_plane_details = []
            self.__file_list_generation = uuid.uuid4()
            self.notify_listeners(URLsRemoved(real_list))

            def undo():
                self.add_urls(real_list, False)

            self.__undo_stack.append((undo, "Remove images"))

    def clear_urls(self, add_undo=True):
        """Remove all URLs from the pipeline"""
        old_urls = list(self.__file_list)
        self.__file_list = []
        if len(old_urls):
            self.__filtered_file_list_images_settings = None
            self.__image_plane_details_metadata_settings = None
            self.__image_plane_details = []
            self.notify_listeners(URLsCleared())
            if add_undo:
                def undo():
                    self.add_urls(old_urls, False)
                self.__undo_stack.append((undo, "Remove images"))

    def load_file_list(self, workspace):
        """Load the pipeline's file_list from the workspace file list

        """
        file_list = workspace.file_list
        if self.__file_list_generation == file_list.generation:
            return
        try:
            urls, metadata, series_names = file_list.get_filelist(want_metadata=True)
        except Exception as instance:
            LOGGER.error("Failed to get file list from workspace", exc_info=True)
            x = IPDLoadException("Failed to get file list from workspace")
            self.notify_listeners(x)
            if x.cancel_run:
                raise instance
        if urls:
            self.start_undoable_action()
            self.clear_urls()
            self.add_urls(urls, metadata=metadata, series_names=series_names)
            self.stop_undoable_action(name="Load file list")
            self.__image_plane_details_generation = file_list.generation
        self.__filtered_image_plane_details_images_settings = tuple()
        self.__filtered_image_plane_details_metadata_settings = tuple()

    def read_file_list(self, path_or_fd, add_undo=True):
        """Read a file of one file or URL per line into the file list

        path - a path to a file or a URL
        """
        if isinstance(path_or_fd, str):
            from ..constants.image import FILE_SCHEME
            from cellprofiler_core.utilities.pathname import url2pathname

            pathname = path_or_fd

            if pathname.startswith(FILE_SCHEME):
                pathname = url2pathname(pathname)

                with open(pathname, "r", encoding="utf-8") as fd:
                    self.read_file_list(fd, add_undo=add_undo)
            elif any(
                pathname.startswith(protocol)
                for protocol in PASSTHROUGH_SCHEMES
            ):
                with urllib.request.urlopen(pathname) as response:
                    data = response.read().decode("utf-8").splitlines()

                self.read_file_list(data, add_undo=add_undo)
            else:
                with open(pathname, "r", encoding="utf-8") as fd:
                    self.read_file_list(fd, add_undo=add_undo)

            return

        self.add_pathnames_to_file_list(
            list(
                map(
                    (lambda x: x.strip()),
                    list(filter((lambda x: len(x) > 0), path_or_fd)),
                )
            ),
            add_undo=add_undo,
        )

    def add_pathnames_to_file_list(self, pathnames, add_undo=True):
        """Add a sequence of paths or URLs to the file list"""
        from cellprofiler_core.utilities.pathname import pathname2url

        urls = []
        for pathname in pathnames:
            if len(pathname) == 0:
                continue
            if pathname.startswith("file:"):
                urls.append(pathname)
            else:
                urls.append(pathname2url(pathname))
        self.add_urls(urls, add_undo=add_undo)

    def get_module_state(self, module_name_or_module):
        """Return an object representing the state of the named module

        module_name - the name of the module

        returns an object that represents the state of the first instance
        of the named module or None if not in pipeline
        """
        if isinstance(module_name_or_module, str):
            modules = [
                module
                for module in self.modules()
                if module.module_name == module_name_or_module
            ]
            if len(modules) == 0:
                return None
            module = modules[0]
        else:
            module = module_name_or_module
        return tuple([s.unicode_value for s in module.settings()])

    def __prepare_run_module(self, module_name, workspace):
        """Execute "prepare_run" on the first instance of the named module"""
        modules = [
            module for module in self.modules() if module.module_name == module_name
        ]
        if len(modules) == 0:
            return False
        return modules[0].prepare_run(workspace)

    def has_cached_filtered_file_list(self):
        """True if the filtered file list is currently cached"""
        images_settings = self.get_module_state("Images")
        if images_settings is None:
            return False
        return self.__filtered_file_list_images_settings == images_settings

    def get_filtered_file_list(self, workspace):
        """Return the file list as filtered by the Images module

        """
        if not self.has_cached_filtered_file_list():
            self.__image_plane_details_metadata_settings = None
            # Images module should always be the first module.
            # PLEASE never let this be untrue.
            first_module = self.module(1)
            from cellprofiler_core.modules.images import Images
            if isinstance(first_module, Images):
                first_module.filter_file_list(workspace)
        return self.__filtered_file_list

    def has_cached_image_plane_details(self):
        """Return True if we have up-to-date image plane details cached"""
        if not self.has_cached_filtered_file_list():
            return False
        metadata_settings = self.get_module_state("Metadata")
        if metadata_settings is None:
            return False
        return self.__image_plane_details_metadata_settings == metadata_settings

    def get_image_plane_list(self, workspace):
        self.__prepare_run_module("Images", workspace)
        return self.__image_plane_list

    def get_image_plane_details(self, workspace):
        """Return the image plane details with metadata computed

        """
        if self.has_cached_image_plane_details():
            return self.__image_plane_details
        self.__available_metadata_keys = set()
        self.__prepare_run_module("Metadata", workspace)
        return self.__image_plane_details

    def get_available_metadata_keys(self):
        """Get the metadata keys from extraction and their types

        Returns a dictionary of metadata key to measurements COLTYPE
        """
        available_keys = []
        modules = self.modules()
        if modules[0].module_name == "Images":
            available_keys += modules[0].get_metadata_keys()
        if modules[1].module_name == "Metadata":
            available_keys += modules[1].get_metadata_keys()
            available_keys = list(set(available_keys))
            return modules[1].get_data_type(available_keys)
        # Probably using LoadData or someone did a bad thing.
        return {}

    def use_case_insensitive_metadata_matching(self, key):
        """Return TRUE if metadata should be matched without regard to case"""
        modules = [
            module for module in self.modules() if module.module_name == "Metadata"
        ]
        if len(modules) == 0:
            return False
        return modules[0].wants_case_insensitive_matching(key)

    def set_filtered_file_list(self, file_list, module):
        """The Images module calls this to report its list of filtered files"""
        self.__filtered_file_list = file_list
        self.__filtered_file_list_images_settings = self.get_module_state(module)

    def set_image_plane_details(self, ipds, available_metadata_keys, module):
        """The Metadata module calls this to report on the extracted IPDs

        ipds - the image plane details to be fed into NamesAndTypes
        available_metadata_keys - the metadata keys collected during IPD
                                  metadata extraction.
        module - the metadata module that made them (so we can cache based
                 on the module's settings.
        """
        self.__image_plane_details = ipds
        self.__available_metadata_keys = available_metadata_keys
        self.__image_plane_details_metadata_settings = self.get_module_state(module)

    LEGACY_LOAD_MODULES = ["LoadData"]

    def has_legacy_loaders(self):
        return any(m.module_name in self.LEGACY_LOAD_MODULES for m in self.modules())

    def needs_default_image_folder(self):
        """Return True if this pipeline makes use of the default image folder"""
        for module in self.modules():
            if module.needs_default_image_folder(self):
                return True
        return False

    def has_undo(self):
        """True if an undo action can be performed"""
        return len(self.__undo_stack)

    def undo(self):
        """Undo the last action"""
        if len(self.__undo_stack):
            action = self.__undo_stack.pop()[0]
            real_undo_stack = self.__undo_stack
            self.__undo_stack = []
            try:
                action()
            finally:
                self.__undo_stack = real_undo_stack

    def undo_action(self):
        """A user-interpretable string telling the user what the action was"""
        if len(self.__undo_stack) == 0:
            return "Nothing to undo"
        return self.__undo_stack[-1][1]

    def undoable_action(self, name="Composite edit"):
        """Return an object that starts and stops an undoable action

        Use this with the "with" statement to create a scope where all
        actions are collected for undo:

        with pipeline.undoable_action():
            pipeline.add_module(module1)
            pipeline.add_module(module2)
        """

        class UndoableAction:
            def __init__(self, pipeline, name):
                self.pipeline = pipeline
                self.name = name

            def __enter__(self):
                self.pipeline.start_undoable_action()

            def __exit__(self, ttype, value, traceback):
                self.pipeline.stop_undoable_action(name)

        return UndoableAction(self, name)

    def start_undoable_action(self):
        """Start editing the pipeline

        This marks a start of a series of actions which will be undone
        all at once.
        """
        self.__undo_start = len(self.__undo_stack)

    def stop_undoable_action(self, name="Composite edit"):
        """Stop editing the pipeline, combining many actions into one"""
        if len(self.__undo_stack) > self.__undo_start + 1:
            # Only combine if two or more edits
            actions = self.__undo_stack[self.__undo_start :]
            del self.__undo_stack[self.__undo_start :]

            def undo():
                for action, message in reversed(actions):
                    action()

            self.__undo_stack.append((undo, name))

    def modules(self, exclude_disabled=True):
        """Return the list of modules

        exclude_disabled - only return enabled modules if True (default)
        """
        if exclude_disabled:
            return [m for m in self.__modules if m.enabled]
        else:
            return self.__modules

    def module(self, module_num):
        module = self.__modules[module_num - 1]
        assert (
            module.module_num == module_num
        ), "Misnumbered module. Expected %d, got %d" % (module_num, module.module_num)
        return module

    @staticmethod
    def capture_module_settings(module):
        """Capture a module's settings for later undo

        module - module in question

        Return a list of setting values that can be fed into the module's
        set_settings_from_values method to reconstruct the module in its original form.
        """
        return [setting.get_unicode_value() for setting in module.settings()]

    def add_module(self, new_module):
        """Insert a module into the pipeline with the given module #

        Insert a module into the pipeline with the given module #.
        'file_name' - the path to the file containing the variables for the module.
        ModuleNum - the one-based index for the placement of the module in the pipeline
        """
        is_image_set_modification = new_module.is_load_module()
        module_num = new_module.module_num
        idx = module_num - 1
        self.__modules = self.__modules[:idx] + [new_module] + self.__modules[idx:]
        for module, mn in zip(
            self.__modules[idx + 1 :],
            list(range(module_num + 1, len(self.__modules) + 1)),
        ):
            module.module_num = mn
        self.notify_listeners(
            ModuleAdded(module_num, is_image_set_modification=is_image_set_modification)
        )
        self.__settings.insert(idx, self.capture_module_settings(new_module))

        def undo():
            self.remove_module(new_module.module_num)

        self.__undo_stack.append((undo, "Add %s module" % new_module.module_name))

    def remove_module(self, module_num):
        """Remove a module from the pipeline

        Remove a module from the pipeline
        ModuleNum - the one-based index of the module
        """
        idx = module_num - 1
        removed_module = self.__modules[idx]
        is_image_set_modification = removed_module.is_load_module()
        self.__modules = self.__modules[:idx] + self.__modules[idx + 1 :]
        for module in self.__modules[idx:]:
            module.module_num = module.module_num - 1
        self.notify_listeners(
            ModuleRemoved(
                module_num, is_image_set_modification=is_image_set_modification
            )
        )
        del self.__settings[idx]

        def undo():
            self.add_module(removed_module)

        self.__undo_stack.append(
            (undo, "Remove %s module" % removed_module.module_name)
        )

    def edit_module(self, module_num, is_image_set_modification):
        """Notify listeners of a module edit

        """
        idx = module_num - 1
        old_settings = self.__settings[idx]
        module = self.__modules[idx]
        new_settings = self.capture_module_settings(module)
        self.notify_listeners(
            ModuleEdited(
                module_num, is_image_set_modification=is_image_set_modification
            )
        )
        self.__settings[idx] = new_settings
        variable_revision_number = module.variable_revision_number
        module_name = module.module_name

        def undo():
            module = self.__modules[idx]
            module.set_settings_from_values(
                old_settings, variable_revision_number, module_name
            )
            self.notify_listeners(ModuleEdited(module_num))
            self.__settings[idx] = old_settings

        self.__undo_stack.append((undo, "Edited %s" % module_name))

    @property
    def file_list(self):
        return self.__file_list

    @property
    def image_plane_details(self):
        return self.__image_plane_details

    @property
    def image_plane_list(self):
        return self.__image_plane_list

    def set_image_plane_list(self, pl):
        self.__image_plane_list = pl

    def on_walk_completed(self):
        self.notify_listeners(FileWalkEnded())

    def test_valid(self):
        """Throw a ValidationError if the pipeline isn't valid

        """
        for module in self.modules():
            module.test_valid(self)

    def notify_listeners(self, event):
        """Notify listeners of an event that happened to this pipeline

        """
        for listener in self.__listeners:
            listener(self, event)

    def add_listener(self, listener):
        self.__listeners.append(listener)

    def remove_listener(self, listener):
        self.__listeners.remove(listener)

    def report_prepare_run_error(self, module, message):
        """Report an error during prepare_run that prevents image set construction

        module - the module that failed

        message - the message for the user

        Report errors due to misconfiguration, such as no files found.
        """
        event = PrepareRunError(module, message)
        self.notify_listeners(event)

    def is_image_from_file(self, image_name):
        """Return True if any module in the pipeline claims to be
        loading this image name from a file."""
        for module in self.modules():
            if module.is_image_from_file(image_name):
                return True
        return False

    def get_measurement_columns(self, terminating_module=None):
        """Return a sequence describing the measurement columns for this pipeline

        This call returns one element per image or object measurement
        made by each module during image set analysis. The element itself
        is a 3-tuple:
        first entry: either one of the predefined measurement categories,
                     {Image", "Experiment" or "Neighbors" or the name of one
                     of the objects.
        second entry: the measurement name (as would be used in a call
                      to add_measurement)
        third entry: the column data type (for instance, "varchar(255)" or
                     "float")
        fourth entry (optional): attribute dictionary. This tags
                     the column with attributes such as MCA_AVAILABLE_POST_GROUP
                     (column values are only added in post_group).
        """
        hash = self.settings_hash()
        if hash != self.__measurement_column_hash:
            self.__measurement_columns = {}
            self.__measurement_column_hash = hash

        terminating_module_num = (
            sys.maxsize if terminating_module is None else terminating_module.module_num
        )
        if terminating_module_num in self.__measurement_columns:
            return self.__measurement_columns[terminating_module_num]
        columns = [
            (EXPERIMENT, M_PIPELINE, COLTYPE_LONGBLOB,),
            (EXPERIMENT, M_VERSION, COLTYPE_VARCHAR,),
            (EXPERIMENT, M_TIMESTAMP, COLTYPE_VARCHAR,),
            (
                EXPERIMENT,
                M_MODIFICATION_TIMESTAMP,
                COLTYPE_VARCHAR,
                {MCA_AVAILABLE_POST_RUN: True},
            ),
            ("Image", GROUP_NUMBER, COLTYPE_INTEGER,),
            ("Image", GROUP_INDEX, COLTYPE_INTEGER,),
            ("Image", GROUP_LENGTH, COLTYPE_INTEGER,),
        ]
        should_write_columns = True
        for module in self.modules():
            if (
                terminating_module is not None
                and terminating_module_num <= module.module_num
            ):
                break
            columns += module.get_measurement_columns(self)
            if module.should_stop_writing_measurements():
                should_write_columns = False
            if should_write_columns:
                module_error_measurement = "ModuleError_%02d%s" % (
                    module.module_num,
                    module.module_name,
                )
                execution_time_measurement = "ExecutionTime_%02d%s" % (
                    module.module_num,
                    module.module_name,
                )
                columns += [
                    ("Image", module_error_measurement, COLTYPE_INTEGER,),
                    ("Image", execution_time_measurement, COLTYPE_FLOAT,),
                ]
        self.__measurement_columns[terminating_module_num] = columns
        return columns

    def get_object_relationships(self):
        """Return a sequence of five-tuples describing all object relationships

        This returns all relationship categories produced by modules via
        Measurements.add_relate_measurement. The format is:
        [(<module-number>, # the module number of the module that wrote it
          <relationship-name>, # the descriptive name of the relationship
          <object-name-1>, # the subject of the relationship
          <object-name-2>, # the object of the relationship
          <when>)] # cpmeas.MCA_AVAILABLE_{EVERY_CYCLE, POST_GROUP}
        """
        result = []
        for module in self.modules():
            result += [
                (module.module_num, i1, o1, i2, o2)
                for i1, o1, i2, o2 in module.get_object_relationships(self)
            ]
        return result

    def get_provider_dictionary(self, groupname, module=None):
        """Get a dictionary of all providers for a given category

        groupname - the name of the category from cellprofiler_core.settings:
            IMAGE_GROUP for image providers, OBJECT_GROUP for object providers
            or MEASUREMENTS_GROUP for measurement providers.

        module - the module that will subscribe to the names. If None, all
        providers are listed, if a module, only the providers for that module's
        place in the pipeline are listed.

        returns a dictionary where the key is the name and the value is
        a list of tuples of module and setting where the module provides
        the name and the setting is the setting that controls the name (and
        the setting can be None).

        """
        target_module = module
        result = {}
        #
        # Walk through the modules to find subscriber and provider settings
        #
        for module in self.modules():
            if (
                target_module is not None
                and target_module.module_num <= module.module_num
            ):
                break
            #
            # Find "other_providers" - providers that aren't specified
            # by single settings.
            #
            p = module.other_providers(groupname)
            for name in p:
                if (name not in result) or target_module is not None:
                    result[name] = []
                result[name].append((module, None))
            if groupname == "measurementsgroup":
                for c in module.get_measurement_columns(self):
                    object_name, feature_name = c[:2]
                    k = (object_name, feature_name)
                    if (k not in result) or target_module is not None:
                        result[k] = []
                    result[k].append((module, None))
            for setting in module.visible_settings():
                if isinstance(setting, Name,) and setting.get_group() == groupname:
                    name = setting.value
                    if name == "Do not use":
                        continue
                    if name not in result or target_module is not None:
                        result[name] = []
                    result[name].append((module, setting))
        return result

    def get_dependency_graph(self):
        """Create a graph that describes the producers and consumers of objects

        returns a list of Dependency objects. These can be used to create a
        directed graph that describes object and image dependencies.
        """
        #
        # These dictionaries have the following structure:
        # * top level dictionary key indicates whether it is an object, image
        #   or measurement dependency
        # * second level dictionary key is the name of the object or image or
        #   a tuple of (object_name, feature) for a measurement.
        # * the value of the second-level dictionary is a list of tuples
        #   where the first element of the tuple is the module and the
        #   second is either None or the setting.
        #
        all_groups = ("objectgroup", "imagegroup", "measurementsgroup")
        providers = dict([(g, self.get_provider_dictionary(g)) for g in all_groups])
        #
        # Now match subscribers against providers.
        #
        result = []
        for module in self.modules():
            for setting in module.visible_settings():
                if isinstance(setting, Name):
                    group = setting.get_group()
                    name = setting.value
                    if group in providers and name in providers[group]:
                        for pmodule, psetting in providers[group][name]:
                            if pmodule.module_num < module.module_num:
                                if group == "objectgroup":
                                    dependency = ObjectDependency(
                                        pmodule, module, name, psetting, setting
                                    )
                                    result.append(dependency)
                                elif group == "imagegroup":
                                    dependency = ImageDependency(
                                        pmodule, module, name, psetting, setting
                                    )
                                    result.append(dependency)
                                break
                elif isinstance(setting, Measurement):
                    object_name = setting.get_measurement_object()
                    feature_name = setting.value
                    key = (object_name, feature_name)
                    if key in providers["measurementsgroup"]:
                        for pmodule, psetting in providers["measurementsgroup"][key]:
                            if pmodule.module_num < module.module_num:
                                dependency = MeasurementDependency(
                                    pmodule,
                                    module,
                                    object_name,
                                    feature_name,
                                    psetting,
                                    setting,
                                )

                                result.append(dependency)
                                break
        return result

    def loaders_settings_hash(self):
        """Return a hash for the settings that control image loading, or None
        for legacy pipelines (which can't be hashed)
        """

        # legacy pipelines can't be cached, because they can load from the
        # Default Image or Output directories.  We could fix this by including
        # them in the hash we use for naming the cache.
        if self.has_legacy_loaders():
            return None

        assert "Groups" in [m.module_name for m in self.modules()]
        return self.settings_hash(until_module="Groups", as_string=True)


def readline(fd):
    """Read a line from fd"""
    try:
        line = next(fd)
        if line is None:
            return None
        line = line.strip()
        return line
    except StopIteration:
        return None
