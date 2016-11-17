import cellprofiler.pipeline

FMT_NATIVE = "Native"


class Pipeline(cellprofiler.pipeline.Pipeline):
    def save(self, fd_or_filename, format=FMT_NATIVE, save_image_plane_details=True):
        super(Pipeline, self).save(fd_or_filename, format, save_image_plane_details)
