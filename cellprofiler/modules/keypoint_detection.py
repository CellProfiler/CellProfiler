"""
KeypointDetection
=================

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

"""

import matplotlib.pyplot
import numpy
import skimage.feature
from cellprofiler_core.module import Module
from cellprofiler_core.object import Objects
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Float
from cellprofiler_core.setting.text import Integer
from cellprofiler_core.setting.text import LabelName
import IPython


class KeypointDetection(Module):
    category = "Advanced"

    module_name = "KeypointDetection"

    variable_revision_number = 1

    def create_settings(self):
        self.x = ImageSubscriber(
            doc="""
            Reference image
            """,
            text="Reference image",
        )

        self.y = ImageSubscriber(
            doc="""
            Image
            """,
            text="Image",
        )

        self.tracks = LabelName(
            doc="""
            Tracks
            """,
            text="Tracks",
            value="tracks",
        )

        self.k = Integer(
            doc="""
            Number of keypoints to detect. The module returns the best 
            keypoints per the Harris corner response if more than *k* 
            `keypoints` are detected. If not, the module returns all detected 
            keypoints.
            """,
            text="k",
            value=512,
        )

        self.maximum_distance = Float(
            doc="""
            Maximum permitted distance between descriptors of two keypoints to 
            be considered a track.
            """,
            text="Maximum distance",
            value=numpy.inf,
        )

    def settings(self):
        return [self.x, self.y, self.tracks, self.k, self.maximum_distance]

    def run(self, workspace):
        detector = skimage.feature.ORB(n_keypoints=self.k.value)

        images = workspace.image_set

        x_image = images.get_image(self.x.value)
        y_image = images.get_image(self.y.value)

        detector.detect_and_extract(x_image.pixel_data)

        x_descriptors = detector.descriptors
        x_keypoints = detector.keypoints

        detector.detect_and_extract(y_image.pixel_data)

        y_descriptors = detector.descriptors
        y_keypoints = detector.keypoints

        matches = skimage.feature.match_descriptors(
            x_descriptors, y_descriptors, max_distance=self.maximum_distance.value
        )

        x_tracks = Objects()
        y_tracks = Objects()

        x_tracks.parent_image = x_image
        y_tracks.parent_image = y_image

        x_indices = numpy.zeros((matches.shape[0], 1))
        y_indices = numpy.zeros((matches.shape[0], 1))

        IPython.embed()

        x_ijv = numpy.concatenate([x_keypoints[matches[:, 0]], x_indices], -1)
        y_ijv = numpy.concatenate([y_keypoints[matches[:, 0]], y_indices], -1)

        x_tracks.set_ijv(x_ijv)
        y_tracks.set_ijv(y_ijv)

        if self.show_window:
            workspace.display_data.x = x_image.pixel_data
            workspace.display_data.y = y_image.pixel_data

            workspace.display_data.x_k = x_keypoints
            workspace.display_data.y_k = y_keypoints

            workspace.display_data.tracks = matches

    def display(self, workspace, figure):
        figure, axes = matplotlib.pyplot.subplots(nrows=1, ncols=1)

        skimage.feature.plot_matches(
            axes[0],
            workspace.display_data.x,
            workspace.display_data.y,
            workspace.display_data.x_k,
            workspace.display_data.y_k,
            workspace.display_data.tracks,
        )

        axes[0].axis("off")
