import io
import re
import os
import tempfile
import urllib.parse
import urllib.request

import nucleus.measurement
import nucleus.pipeline


def dump(pipeline, fp, save_image_plane_details):
    if len(pipeline.file_list) == 0:
        save_image_plane_details = False

    date_revision = int(re.sub(r"\.|rc\d", "", nucleus.__version__))
    module_count = len(pipeline.modules())

    fp.write("CellProfiler Pipeline: http://www.cellprofiler.org\n")
    fp.write(f"Version:{nucleus.pipeline.NATIVE_VERSION:d}\n")
    fp.write(f"DateRevision:{date_revision:d}\n")
    fp.write(f"GitHash:{''}\n")
    fp.write(f"ModuleCount:{module_count:d}\n")
    fp.write(f"HasImagePlaneDetails:{save_image_plane_details}\n")

    default_module_attributes = (
        "module_num",
        "svn_version",
        "variable_revision_number",
        "show_window",
        "notes",
        "batch_state",
        "enabled",
        "wants_pause",
    )

    for module in pipeline.modules():
        fp.write("\n")

        module_attributes = []

        for default_module_attribute in default_module_attributes:
            attribute = repr(getattr(module, default_module_attribute))

            module_attributes += [f"{default_module_attribute}:{attribute}"]

        fp.write(f"{module.module_name}:[{'|'.join(module_attributes)}]\n")

        for setting in module.settings():
            fp.write(f"    {setting.text}:{setting.unicode_value}\n")

    if save_image_plane_details:
        fp.write("\n")

        nucleus.pipeline.write_file_list(fp, pipeline.file_list)


def load(pipeline, fd_or_filename):
    """Load the pipeline from a file

    fd_or_filename - either the name of a file or a file-like object
    """
    pipeline.__modules = []
    pipeline.__undo_stack = []
    pipeline.__undo_start = None

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

        needs_close = True

        filename = fd_or_filename
    else:
        # Assume is string URL
        parsed_path = urllib.parse.urlparse(fd_or_filename)

        if len(parsed_path.scheme) < 2:
            raise IOError("Could not find file, " + fd_or_filename)

        fd = six.moves.urllib.request.urlopen(fd_or_filename)

        return pipeline.load(fd)

    if nucleus.pipeline.Pipeline.is_pipeline_txt_fd(fd):
        pipeline.loadtxt(fd)

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
        load(pipeline, filename)
        os.unlink(filename)
        return
    else:
        m = nucleus.measurement.load_measurements(filename)
        pipeline_text = m.get_experiment_measurement(
            nucleus.pipeline.M_PIPELINE
        )
        pipeline_text = pipeline_text
        load(pipeline, io.StringIO(pipeline_text))
        return
