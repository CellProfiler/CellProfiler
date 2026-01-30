import centrosome.zernike
import matplotlib.cm
import numpy
import numpy.ma
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.preferences import get_default_colormap
from cellprofiler_core.setting import (
    HiddenCount,
    Divider,
    SettingsGroup,
    Binary,
    ValidationError,
)
from cellprofiler_core.setting.choice import Choice, Colormap
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.subscriber import (
    LabelSubscriber,
    ImageListSubscriber,
    ImageSubscriber,
)
from cellprofiler_core.setting.text import Integer, ImageName
from cellprofiler_core.utilities.core.object import crop_labels_and_image
from cellprofiler_library.opts.measureobjectintensitydistribution import (
    CenterChoice,
    IntensityZernike,
    Feature, 
    FullFeature,
    MeasurementFeature,
    OverflowFeature,
    C_ALL,
    Z_ALL,
    M_CATEGORY,
    F_ALL,
    MEASUREMENT_CHOICES, 
    MEASUREMENT_ALIASES,
    FF_SCALE,
    FF_GENERIC
)
from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.modules._measureobjectintensitydistribution import (
    measure_intensity_distribution,
    measure_intensity_zernikes,
    compute_zernike_geometry,
    compute_radial_distribution_geometry
)
from cellprofiler_library.functions.segmentation import convert_label_set_to_ijv, indices_from_ijv
import cellprofiler.gui.help.content
MeasureObjectIntensityDistribution_Magnitude_Phase = cellprofiler.gui.help.content.image_resource(
    "MeasureObjectIntensityDistribution_Magnitude_Phase.png"
)
MeasureObjectIntensityDistribution_Edges_Centers = cellprofiler.gui.help.content.image_resource(
    "MeasureObjectIntensityDistribution_Edges_Centers.png"
)

__doc__ = """
MeasureObjectIntensityDistribution
==================================

**MeasureObjectIntensityDistribution** measures the spatial distribution of
intensities within each object.

Given an image with objects identified, this module measures the
intensity distribution from each object’s center to its boundary within
a set of bins, i.e., rings that you specify.

|MeasureObjectIntensityDistribution_image0|

The distribution is measured from the center of the object, where the
center is defined as the point farthest from any edge. The numbering of bins is
from 1 (innermost) to *N* (outermost), where *N* is the number of bins
you specify. Alternatively, if primary objects exist within
the object of interest (e.g., nuclei within cells), you can choose the
center of the primary objects as the center from which to measure the
radial distribution. This might be useful in cytoplasm-to-nucleus
translocation experiments, for example. Note that the ring widths are
normalized per-object, i.e., not necessarily a constant width across
objects.

|MeasureObjectIntensityDistribution_image1|

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also **MeasureObjectIntensity** and **MeasureTexture**.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *FracAtD:* Fraction of total stain in an object at a given radius.
-  *MeanFrac:* Mean fractional intensity at a given radius; calculated
   as fraction of total intensity normalized by fraction of pixels at a
   given radius.
-  *RadialCV:* Coefficient of variation of intensity within a ring,
   calculated across 8 slices.
-  *Zernike:* The Zernike features characterize the distribution of
   intensity across the object. For instance, Zernike 1,1 has a high
   value if the intensity is low on one side of the object and high on
   the other. The ZernikeMagnitude feature records the rotationally
   invariant degree magnitude of the moment and the ZernikePhase feature
   gives the moment’s orientation.

.. |MeasureObjectIntensityDistribution_image0| image:: {MeasureObjectIntensityDistribution_Magnitude_Phase}
.. |MeasureObjectIntensityDistribution_image1| image:: {MeasureObjectIntensityDistribution_Edges_Centers}

""".format(
    **{
        "MeasureObjectIntensityDistribution_Magnitude_Phase": MeasureObjectIntensityDistribution_Magnitude_Phase,
        "MeasureObjectIntensityDistribution_Edges_Centers": MeasureObjectIntensityDistribution_Edges_Centers,
    }
)

"""# of settings aside from groups"""
SETTINGS_STATIC_COUNT = 3
"""# of settings in image group"""
SETTINGS_IMAGE_GROUP_COUNT = 1
"""# of settings in object group"""
SETTINGS_OBJECT_GROUP_COUNT = 3
"""# of settings in bin group, v1"""
SETTINGS_BIN_GROUP_COUNT_V1 = 1
"""# of settings in bin group, v2"""
SETTINGS_BIN_GROUP_COUNT_V2 = 3
SETTINGS_BIN_GROUP_COUNT = 3
"""# of settings in heatmap group, v4"""
SETTINGS_HEATMAP_GROUP_COUNT_V4 = 7
SETTINGS_HEATMAP_GROUP_COUNT = 7
"""Offset of center choice in object group"""
SETTINGS_CENTER_CHOICE_OFFSET = 1

class MeasureObjectIntensityDistribution(Module):
    module_name = "MeasureObjectIntensityDistribution"
    category = "Measurement"
    variable_revision_number = 6

    def create_settings(self):
        self.images_list = ImageListSubscriber(
            "Select images to measure",
            [],
            doc="""Select the images whose intensity distribution you want to measure.""",
        )

        self.objects = []
        self.bin_counts = []
        self.heatmaps = []
        self.object_count = HiddenCount(self.objects)
        self.bin_counts_count = HiddenCount(self.bin_counts)
        self.heatmap_count = HiddenCount(self.heatmaps)
        self.wants_zernikes = Choice(
            "Calculate intensity Zernikes?",
            Z_ALL,
            doc="""\
This setting determines whether the intensity Zernike moments are
calculated. Choose *{Z_NONE}* to save computation time by not
calculating the Zernike moments. Choose *{Z_MAGNITUDES}* to only save
the magnitude information and discard information related to the
object’s angular orientation. Choose *{Z_MAGNITUDES_AND_PHASE}* to
save the phase information as well. The last option lets you recover
each object’s rough appearance from the Zernikes but may not contribute
useful information for classifying phenotypes.

|MeasureObjectIntensityDistribution_image0|

.. |MeasureObjectIntensityDistribution_image0| image:: {MeasureObjectIntensityDistribution_Magnitude_Phase}
""".format(
                **{
                    "Z_NONE": IntensityZernike.NONE.value,
                    "Z_MAGNITUDES": IntensityZernike.MAGNITUDES.value,
                    "Z_MAGNITUDES_AND_PHASE": IntensityZernike.MAGNITUDES_AND_PHASE.value,
                    "MeasureObjectIntensityDistribution_Magnitude_Phase": MeasureObjectIntensityDistribution_Magnitude_Phase,
                }
            ),
        )

        self.zernike_degree = Integer(
            "Maximum zernike moment",
            value=9,
            minval=1,
            maxval=20,
            doc="""\
(*Only if "{wants_zernikes}" is "{Z_MAGNITUDES}" or "{Z_MAGNITUDES_AND_PHASE}"*)

This is the maximum radial moment that will be calculated. There are
increasing numbers of azimuthal moments as you increase the radial
moment, so higher values are increasingly expensive to calculate.
""".format(
                **{
                    "wants_zernikes": self.wants_zernikes.text,
                    "Z_MAGNITUDES": IntensityZernike.MAGNITUDES.value,
                    "Z_MAGNITUDES_AND_PHASE": IntensityZernike.MAGNITUDES_AND_PHASE.value,
                }
            ),
        )

        self.spacer_1 = Divider()

        self.add_object_button = DoSomething("", "Add another object", self.add_object)

        self.spacer_2 = Divider()

        self.add_bin_count_button = DoSomething(
            "", "Add another set of bins", self.add_bin_count
        )

        self.spacer_3 = Divider()

        self.add_heatmap_button = DoSomething(
            "",
            "Add another heatmap display",
            self.add_heatmap,
            doc="""\
Press this button to add a display of one of the radial distribution
measurements. Each radial band of the object is colored using a
heatmap according to the measurement value for that band.
""",
        )

        self.add_object(can_remove=False)

        self.add_bin_count(can_remove=False)

    def add_object(self, can_remove=True):
        group = SettingsGroup()

        if can_remove:
            group.append("divider", Divider(line=False))

        group.append(
            "object_name",
            LabelSubscriber(
                "Select objects to measure",
                "None",
                doc="Select the objects whose intensity distribution you want to measure.",
            ),
        )

        group.append(
            "center_choice",
            Choice(
                "Object to use as center?",
                C_ALL,
                doc="""\
There are three ways to specify the center of the radial measurement:

-  *{C_SELF}:* Use the centers of these objects for the radial
   measurement.
-  *{C_CENTERS_OF_OTHER}:* Use the centers of other objects for the
   radial measurement.
-  *{C_EDGES_OF_OTHER}:* Measure distances from the edge of the other
   object to each pixel outside of the centering object. Do not include
   pixels within the centering object in the radial measurement
   calculations.

For example, if measuring the radial distribution in a Cell object, you
can use the center of the Cell objects (*{C_SELF}*) or you can use
previously identified Nuclei objects as the centers
(*{C_CENTERS_OF_OTHER}*).

|MeasureObjectIntensityDistribution_image1|

.. |MeasureObjectIntensityDistribution_image1| image:: {MeasureObjectIntensityDistribution_Edges_Centers}
""".format(
                    **{
                        "C_SELF": CenterChoice.SELF.value,
                        "C_CENTERS_OF_OTHER": CenterChoice.CENTERS_OF_OTHER.value,
                        "C_EDGES_OF_OTHER": CenterChoice.EDGES_OF_OTHER.value,
                        "MeasureObjectIntensityDistribution_Edges_Centers": MeasureObjectIntensityDistribution_Edges_Centers,
                    }
                ),
            ),
        )

        group.append(
            "center_object_name",
            LabelSubscriber(
                "Select objects to use as centers",
                "None",
                doc="""\
*(Used only if “{C_CENTERS_OF_OTHER}” are selected for centers)*

Select the object to use as the center, or select *None* to use the
input object centers (which is the same as selecting *{C_SELF}* for the
object centers).
""".format(
                    **{"C_CENTERS_OF_OTHER": CenterChoice.CENTERS_OF_OTHER.value, "C_SELF": CenterChoice.SELF.value}
                ),
            ),
        )

        if can_remove:
            group.append(
                "remover",
                RemoveSettingButton("", "Remove this object", self.objects, group),
            )

        self.objects.append(group)

    def add_bin_count(self, can_remove=True):
        group = SettingsGroup()

        if can_remove:
            group.append("divider", Divider(line=False))

        group.append(
            "wants_scaled",
            Binary(
                "Scale the bins?",
                True,
                doc="""\
Select *{YES}* to divide the object radially into the number of bins
that you specify.

Select *{NO}* to create the number of bins you specify based on
distance. For this option, you will be asked to specify a maximum
distance so that each object will have the same measurements (which
might be zero for small objects) and so that the measurements can be
taken without knowing the maximum object radius before the run starts.
""".format(
                    **{"YES": "Yes", "NO": "No"}
                ),
            ),
        )

        group.append(
            "bin_count",
            Integer(
                "Number of bins",
                4,
                2,
                doc="""\
Specify the number of bins that you want to use to measure the
distribution. Radial distribution is measured with respect to a series
of concentric rings starting from the object center (or more generally,
between contours at a normalized distance from the object center). This
number specifies the number of rings into which the distribution is to
be divided. Additional ring counts can be specified by clicking the *Add
another set of bins* button.""",
            ),
        )

        group.append(
            "maximum_radius",
            Integer(
                "Maximum radius",
                100,
                minval=1,
                doc="""\
Specify the maximum radius for the unscaled bins. The unscaled binning method creates the number of
bins that you specify and creates equally spaced bin boundaries up to the maximum radius. Parts of
the object that are beyond this radius will be counted in an overflow bin. The radius is measured
in pixels.
""",
            ),
        )

        group.can_remove = can_remove

        if can_remove:
            group.append(
                "remover",
                RemoveSettingButton(
                    "", "Remove this set of bins", self.bin_counts, group
                ),
            )

        self.bin_counts.append(group)

    def get_bin_count_choices(self, pipeline=None):
        choices = []
        for bin_count in self.bin_counts:
            nbins = str(bin_count.bin_count.value)
            if nbins != choices:
                choices.append(nbins)
        return choices

    def add_heatmap(self):
        group = SettingsGroup()

        if len(self.heatmaps) > 0:
            group.append("divider", Divider(line=False))

        group.append(
            "image_name",
            MORDImageNameSubscriber(
                "Image",
                doc="""\
The heatmap will be displayed with measurements taken using this image. The setting will let you
choose from among the images you have specified in "Select image to measure".
""",
            ),
        )

        group.image_name.set_module(self)

        group.append(
            "object_name",
            MORDObjectNameSubscriber(
                "Objects to display",
                doc="""\
The objects to display in the heatmap. You can select any of the
objects chosen in "Select objects to measure".""",
            ),
        )

        group.object_name.set_module(self)

        group.append(
            "bin_count",
            Choice(
                "Number of bins",
                self.get_bin_count_choices(),
                choices_fn=self.get_bin_count_choices,
            ),
        )

        def get_number_of_bins(module=self, group=group):
            if len(module.bin_counts) == 1:
                return module.bin_counts[0].bin_count.value

            return int(group.bin_count.value)

        group.get_number_of_bins = get_number_of_bins

        group.append(
            "measurement",
            Choice(
                "Measurement", MEASUREMENT_CHOICES, doc="The measurement to display."
            ),
        )

        group.append(
            "colormap",
            Colormap(
                "Color map",
                value="Blues",
                doc="""\
The color map setting chooses the color palette that will be
used to render the different values for your measurement. If you
choose "gray", the image will label each of the bins with the
actual image measurement.""",
            ),
        )

        group.append(
            "wants_to_save_display",
            Binary(
                "Save display as image?",
                False,
                doc="""\
This setting allows you to save the heatmap display as an image that can
be output using the **SaveImages** module. Choose *{YES}* to save the
display or *{NO}* if the display is not needed.
""".format(
                    **{"YES": "Yes", "NO": "No"}
                ),
            ),
        )

        group.append(
            "display_name",
            ImageName(
                "Output image name",
                "Heatmap",
                doc="""\
*(Only used if “Save display as image?” is “{YES}”)*

This setting names the heatmap image so that the name you enter here can
be selected in a later **SaveImages** or other module.
""".format(
                    **{"YES": "Yes"}
                ),
            ),
        )

        group.append(
            "remover",
            RemoveSettingButton(
                "", "Remove this heatmap display", self.heatmaps, group
            ),
        )

        self.heatmaps.append(group)

    def validate_module(self, pipeline):
        images = set()
        if len(self.images_list.value) == 0:
            raise ValidationError("No images selected", self.images_list)
        for image_name in self.images_list.value:
            if image_name in images:
                raise ValidationError(
                    "%s has already been selected" % image_name, image_name
                )
            images.add(image_name)

        objects = set()
        for group in self.objects:
            if group.object_name.value in objects:
                raise ValidationError(
                    "{} has already been selected".format(group.object_name.value),
                    group.object_name,
                )
            objects.add(group.object_name.value)

        bins = set()
        for group in self.bin_counts:
            if group.bin_count.value in bins:
                raise ValidationError(
                    "{} has already been selected".format(group.bin_count.value),
                    group.bin_count,
                )
            bins.add(group.bin_count.value)

    def settings(self):
        result = [
            self.images_list,
            self.object_count,
            self.bin_counts_count,
            self.heatmap_count,
            self.wants_zernikes,
            self.zernike_degree,
        ]

        for x in (self.objects, self.bin_counts, self.heatmaps):
            for settings in x:
                temp = settings.pipeline_settings()
                result += temp

        return result

    def visible_settings(self):
        result = [self.wants_zernikes]

        if self.wants_zernikes != IntensityZernike.NONE.value:
            result.append(self.zernike_degree)

        result += [self.images_list, self.spacer_1]

        for settings in self.objects:
            temp = settings.visible_settings()

            if settings.center_choice.value == CenterChoice.SELF.value:
                temp.remove(settings.center_object_name)

            result += temp

        result += [self.add_object_button, self.spacer_2]

        for settings in self.bin_counts:
            result += [settings.wants_scaled, settings.bin_count]

            if not settings.wants_scaled:
                result += [settings.maximum_radius]

            if settings.can_remove:
                result += [settings.remover]

        result += [self.add_bin_count_button, self.spacer_3]

        for settings in self.heatmaps:
            if hasattr(settings, "divider"):
                result.append(settings.divider)

            if settings.image_name.is_visible():
                result.append(settings.image_name)

            if settings.object_name.is_visible():
                result.append(settings.object_name)

            if len(self.bin_counts) > 1:
                result.append(settings.bin_count)

            result += [
                settings.measurement,
                settings.colormap,
                settings.wants_to_save_display,
            ]

            if settings.wants_to_save_display:
                result.append(settings.display_name)

            result.append(settings.remover)

        result += [self.add_heatmap_button]

        return result

    def prepare_settings(self, setting_values):
        objects_count, bin_counts_count, heatmap_count = [
            int(x) for x in setting_values[1:4]
        ]

        for sequence, add_fn, count in (
            (self.objects, self.add_object, objects_count),
            (self.bin_counts, self.add_bin_count, bin_counts_count),
            (self.heatmaps, self.add_heatmap, heatmap_count),
        ):
            while len(sequence) > count:
                del sequence[-1]

            while len(sequence) < count:
                add_fn()

    def run(self, workspace):
        header = (
            "Image",
            "Objects",
            "Bin # (innermost=1)",
            "Bin count",
            "Fraction",
            "Intensity",
            "COV",
        )

        stats = []

        d = {}

        for object_name_setting in self.objects:
             object_name = object_name_setting.object_name.value
             objects = workspace.object_set.get_objects(object_name)
             labels = objects.segmented
             full_shape = labels.shape
             
             center_choice = object_name_setting.center_choice.value
             center_object_name = (
                 object_name_setting.center_object_name.value 
                 if center_choice != CenterChoice.SELF.value 
                 else None
             )
             
             center_object_labels = None
             if center_object_name:
                 center_object_labels = workspace.object_set.get_objects(center_object_name).segmented

             # Calculate bounding box for cropping
             if numpy.any(labels > 0):
                 where = numpy.argwhere(labels > 0)
                 (y_min, x_min), (y_max, x_max) = where.min(0), where.max(0) + 1
                 # Slices covering the objects with padding
                 obj_slices = (slice(max(0, y_min - 1), y_max + 1), slice(max(0, x_min - 1), x_max + 1))
             else:
                 obj_slices = (slice(0, 0), slice(0, 0))
             
             nobjects_full = numpy.max(labels) if labels.size > 0 else 0

             # Cache for radial geometry: (wants_scaled, max_radius) -> (geometry, objects_indices_cropped)
             radial_geometry_cache = {}

             for bin_count_settings in self.bin_counts:
                 bin_count = bin_count_settings.bin_count.value
                 wants_scaled = bin_count_settings.wants_scaled.value
                 maximum_radius = bin_count_settings.maximum_radius.value
                 
                 geom_key = (wants_scaled, maximum_radius)
                 
                 for image_name in self.images_list.value:
                     image = workspace.image_set.get_image(image_name, must_be_grayscale=True)
                     im_shape = image.pixel_data.shape
                     
                     # Calculate intersection crop
                     if obj_slices[0].start == obj_slices[0].stop:
                         img_slices = obj_slices
                     else:
                         y_start = max(0, min(obj_slices[0].start, im_shape[0], full_shape[0]))
                         y_stop = max(0, min(obj_slices[0].stop, im_shape[0], full_shape[0]))
                         x_start = max(0, min(obj_slices[1].start, im_shape[1], full_shape[1]))
                         x_stop = max(0, min(obj_slices[1].stop, im_shape[1], full_shape[1]))
                         img_slices = (slice(y_start, y_stop), slice(x_start, x_stop))
                     
                     if img_slices[0].start == img_slices[0].stop or img_slices[1].start == img_slices[1].stop:
                         labels_cropped = numpy.zeros((0, 0), dtype=labels.dtype)
                         pixel_data_cropped = numpy.zeros((0, 0), dtype=image.pixel_data.dtype)
                     else:
                         labels_cropped = labels[img_slices]
                         pixel_data_cropped = image.pixel_data[img_slices]
                     
                     nobjects = nobjects_full
                         
                     # Heatmaps
                     heatmap_features = []
                     relevant_heatmaps = []
                     for heatmap in self.heatmaps:
                         if (heatmap.object_name.get_objects_name() == object_name and
                             heatmap.image_name.get_image_name() == image_name and
                             heatmap.get_number_of_bins() == bin_count):
                                 feature = MEASUREMENT_ALIASES.get(heatmap.measurement.value, heatmap.measurement.value)
                                 heatmap_features.append(feature)
                                 relevant_heatmaps.append((heatmap, feature))
                     
                     # Check/Update Cache
                     # Note: geometry depends on labels_cropped.
                     # If image sizes vary, img_slices vary, labels_cropped vary -> geometry varies.
                     # We only cache if we assume labels_cropped is constant.
                     # If image sizes vary, we might have misses or incorrect reuse?
                     # Standard CP assumes images in a run are same size.
                     # If they are, img_slices is constant.
                     # So caching is safe.
                     
                     if geom_key not in radial_geometry_cache and nobjects > 0:
                         objects_ijv_cropped = convert_label_set_to_ijv(labels_cropped)
                         objects_indices_cropped = indices_from_ijv(objects_ijv_cropped)
                         
                         center_labels_cropped = None
                         if center_object_labels is not None:
                             center_labels_cropped = center_object_labels[img_slices]
                         
                         radial_geometry_cache[geom_key] = (
                             compute_radial_distribution_geometry(
                                 labels_cropped, 
                                 center_labels_cropped, 
                                 objects_indices_cropped, 
                                 center_choice, 
                                 wants_scaled, 
                                 maximum_radius
                             ),
                             objects_indices_cropped
                         )
                     
                     geometry = None
                     objects_indices_cropped = None
                     if nobjects > 0 and geom_key in radial_geometry_cache:
                         geometry, objects_indices_cropped = radial_geometry_cache[geom_key]
                     
                     center_labels_cropped = None
                     if center_object_labels is not None:
                         center_labels_cropped = center_object_labels[img_slices]

                     lib_measurements, iter_stats, heatmap_data = measure_intensity_distribution(
                         pixel_data_cropped,
                         image_name,
                         object_name,
                         labels_cropped,
                         nobjects,
                         bin_count,
                         wants_scaled,
                         maximum_radius,
                         center_choice,
                         center_object_name,
                         center_labels_cropped,
                         geometry=geometry,
                         objects_indices=objects_indices_cropped,
                         heatmap_features=heatmap_features if heatmap_features else None,
                         return_visualization_data=True
                     )
                     
                     # Unpack measurements
                     for obj_name, features in lib_measurements.objects.items():
                         for feature_name, val in features.items():
                             workspace.measurements.add_measurement(obj_name, feature_name, val)
                             
                     stats += iter_stats
                     
                     # Store heatmaps (Uncropped)
                     for heatmap, feature in relevant_heatmaps:
                         if feature in heatmap_data:
                             full_heatmap = numpy.zeros(full_shape)
                             full_heatmap[img_slices] = heatmap_data[feature]
                             d[id(heatmap)] = full_heatmap

        if self.wants_zernikes != IntensityZernike.NONE.value:
            zernike_opts = self.wants_zernikes.value
            zernike_degree = self.zernike_degree.value
            
            zernike_geometry_cache = {} 
            
            for object_name_setting in self.objects:
                object_name = object_name_setting.object_name.value
                objects = workspace.object_set.get_objects(object_name)
                # Use get_labels() for Zernike geometry (List of (small, ind))
                labels_set = objects.get_labels()
                
                if (object_name, zernike_degree) not in zernike_geometry_cache:
                    zernike_geometry_cache[(object_name, zernike_degree)] = compute_zernike_geometry(
                        labels_set, zernike_degree
                    )
                geometry = zernike_geometry_cache[(object_name, zernike_degree)]
                
                for image_name in self.images_list.value:
                     image = workspace.image_set.get_image(image_name, must_be_grayscale=True)
                     
                     lib_measurements = measure_intensity_zernikes(
                         image.pixel_data,
                         image.mask,
                         image_name,
                         object_name,
                         labels_set,
                         zernike_degree,
                         zernike_opts,
                         geometry=geometry
                     )
                     
                     for obj_name, features in lib_measurements.objects.items():
                         for feature_name, val in features.items():
                             workspace.measurements.add_measurement(obj_name, feature_name, val)

        if self.show_window:
            workspace.display_data.header = header

            workspace.display_data.stats = stats

            workspace.display_data.heatmaps = []

        for heatmap in self.heatmaps:
            heatmap_img = d.get(id(heatmap))

            if heatmap_img is not None:
                if self.show_window or heatmap.wants_to_save_display:
                    labels = workspace.object_set.get_objects(
                        heatmap.object_name.get_objects_name()
                    ).segmented

                if self.show_window:
                    workspace.display_data.heatmaps.append((heatmap_img, labels != 0))

                if heatmap.wants_to_save_display:
                    colormap = heatmap.colormap.value

                    if colormap == matplotlib.cm.gray.name:
                        output_pixels = heatmap_img
                    else:
                        if colormap == "Default":
                            colormap = get_default_colormap()

                        cm = matplotlib.cm.ScalarMappable(cmap=colormap)

                        output_pixels = cm.to_rgba(heatmap_img)[:, :, :3]

                        output_pixels[labels == 0, :] = 0

                    parent_image = workspace.image_set.get_image(
                        heatmap.image_name.get_image_name()
                    )

                    output_img = Image(output_pixels, parent_image=parent_image)

                    img_name = heatmap.display_name.value

                    workspace.image_set.add(img_name, output_img)

    def display(self, workspace, figure):
        header = workspace.display_data.header

        stats = workspace.display_data.stats

        n_plots = len(workspace.display_data.heatmaps) + 1

        n_vert = int(numpy.sqrt(n_plots))

        n_horiz = int(numpy.ceil(float(n_plots) / n_vert))

        if len(self.heatmaps) > 0:
            helptext = "short"
        else:
            helptext = "default"

        figure.set_subplots((n_horiz, n_vert))

        figure.subplot_table(0, 0, stats, col_labels=header, title=helptext)

        idx = 1

        sharexy = None

        for heatmap, (heatmap_img, mask) in zip(
            self.heatmaps, workspace.display_data.heatmaps
        ):

            heatmap_img = numpy.ma.array(heatmap_img, mask=~mask)

            if heatmap_img is not None:
                title = "{} {} {}".format(
                    heatmap.image_name.get_image_name(),
                    heatmap.object_name.get_objects_name(),
                    heatmap.measurement.value,
                )

                x = idx % n_horiz

                y = int(idx / n_horiz)

                colormap = heatmap.colormap.value

                if colormap == "Default":
                    colormap = get_default_colormap()

                if sharexy is None:
                    sharexy = figure.subplot_imshow(
                        x,
                        y,
                        heatmap_img,
                        title=title,
                        colormap=colormap,
                        normalize=False,
                        vmin=numpy.min(heatmap_img),
                        vmax=numpy.max(heatmap_img),
                        colorbar=False,
                    )
                else:
                    figure.subplot_imshow(
                        x,
                        y,
                        heatmap_img,
                        title=title,
                        colormap=colormap,
                        colorbar=False,
                        normalize=False,
                        vmin=numpy.min(heatmap_img),
                        vmax=numpy.max(heatmap_img),
                        sharexy=sharexy,
                    )
                idx += 1
                 

    def get_zernike_magnitude_name(self, image_name, n, m):
        """The feature name of the magnitude of a Zernike moment

        image_name - the name of the image being measured
        n - the radial moment of the Zernike
        m - the azimuthal moment of the Zernike
        """
        return "_".join((M_CATEGORY, FullFeature.ZERNIKE_MAGNITUDE.value, image_name, str(n), str(m)))

    def get_zernike_phase_name(self, image_name, n, m):
        """The feature name of the phase of a Zernike moment

        image_name - the name of the image being measured
        n - the radial moment of the Zernike
        m - the azimuthal moment of the Zernike
        """
        return "_".join((M_CATEGORY, FullFeature.ZERNIKE_PHASE.value, image_name, str(n), str(m)))

    def get_measurement_columns(self, pipeline):
        columns = []

        for image_name in self.images_list.value:
            for o in self.objects:
                object_name = o.object_name.value

                for bin_count_obj in self.bin_counts:
                    bin_count = bin_count_obj.bin_count.value

                    wants_scaling = bin_count_obj.wants_scaled.value

                    for feature, ofeature in (
                        (MeasurementFeature.FRAC_AT_D.value, OverflowFeature.FRAC_AT_D.value),
                        (MeasurementFeature.MEAN_FRAC.value, OverflowFeature.MEAN_FRAC.value),
                        (MeasurementFeature.RADIAL_CV.value, OverflowFeature.RADIAL_CV.value),
                    ):
                        for bin in range(1, bin_count + 1):
                            columns.append(
                                (
                                    object_name,
                                    feature % (image_name, bin, bin_count),
                                    COLTYPE_FLOAT,
                                )
                            )

                        if not wants_scaling:
                            columns.append(
                                (object_name, ofeature % image_name, COLTYPE_FLOAT,)
                            )

                    if self.wants_zernikes != IntensityZernike.NONE.value:
                        name_fns = [self.get_zernike_magnitude_name]

                        if self.wants_zernikes == IntensityZernike.MAGNITUDES_AND_PHASE.value:
                            name_fns.append(self.get_zernike_phase_name)

                        max_n = self.zernike_degree.value

                        for name_fn in name_fns:
                            for n, m in centrosome.zernike.get_zernike_indexes(
                                max_n + 1
                            ):
                                ftr = name_fn(image_name, n, m)

                                columns.append((object_name, ftr, COLTYPE_FLOAT,))

        return columns

    def get_categories(self, pipeline, object_name):
        if object_name in [x.object_name.value for x in self.objects]:
            return [M_CATEGORY]

        return []

    def get_measurements(self, pipeline, object_name, category):
        if category in self.get_categories(pipeline, object_name):
            if self.wants_zernikes == IntensityZernike.NONE.value:
                return F_ALL

            if self.wants_zernikes == IntensityZernike.MAGNITUDES.value:
                return F_ALL + [FullFeature.ZERNIKE_MAGNITUDE.value]

            return F_ALL + [FullFeature.ZERNIKE_MAGNITUDE.value, FullFeature.ZERNIKE_PHASE.value]

        return []

    def get_measurement_images(self, pipeline, object_name, category, feature):
        if feature in self.get_measurements(pipeline, object_name, category):
            return self.images_list.value
        return []

    def get_measurement_scales(
        self, pipeline, object_name, category, feature, image_name
    ):
        if image_name in self.get_measurement_images(
            pipeline, object_name, category, feature
        ):
            if feature in (FullFeature.ZERNIKE_MAGNITUDE.value, FullFeature.ZERNIKE_PHASE.value):
                n_max = self.zernike_degree.value

                result = [
                    "{}_{}".format(n, m)
                    for n, m in centrosome.zernike.get_zernike_indexes(n_max + 1)
                ]
            else:
                result = [
                    FF_SCALE % (bin, bin_count.bin_count.value)
                    for bin_count in self.bin_counts
                    for bin in range(1, bin_count.bin_count.value + 1)
                ]

                if any(
                    [not bin_count.wants_scaled.value for bin_count in self.bin_counts]
                ):
                    result += [FullFeature.OVERFLOW.value]

            return result

        return []

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            n_images, n_objects, n_bins = [
                int(setting) for setting in setting_values[:3]
            ]

            off_bins = (
                SETTINGS_STATIC_COUNT
                + n_images * SETTINGS_IMAGE_GROUP_COUNT
                + n_objects * SETTINGS_OBJECT_GROUP_COUNT
            )

            new_setting_values = setting_values[:off_bins]

            for bin_count in setting_values[off_bins:]:
                new_setting_values += ["Yes", bin_count, "100"]

            setting_values = new_setting_values

            variable_revision_number = 2

        if variable_revision_number == 2:
            n_images, n_objects = [int(setting) for setting in setting_values[:2]]

            off_objects = SETTINGS_STATIC_COUNT + n_images * SETTINGS_IMAGE_GROUP_COUNT

            setting_values = list(setting_values)

            for i in range(n_objects):
                offset = (
                    off_objects
                    + i * SETTINGS_OBJECT_GROUP_COUNT
                    + SETTINGS_CENTER_CHOICE_OFFSET
                )

                if setting_values[offset] == "Other objects":
                    setting_values[offset] = CenterChoice.CENTERS_OF_OTHER.value

            variable_revision_number = 3

        if variable_revision_number == 3:
            # added heatmaps
            # Need a heatmap_count = 0
            #
            setting_values = setting_values[:3] + ["0"] + setting_values[3:]

            variable_revision_number = 4

        if variable_revision_number == 4:
            #
            # Added zernikes
            #
            setting_values = setting_values[:4] + ["None", "9"] + setting_values[4:]

            variable_revision_number = 5

        if variable_revision_number == 5:
            n_images = int(setting_values[0])
            mid = setting_values[1:6]
            end = setting_values[6 + n_images :]

            images_set = set(setting_values[6 : 6 + n_images])
            if "None" in images_set:
                images_set.remove("None")
            images_string = ", ".join(map(str, images_set))

            setting_values = [images_string] + mid + end

            variable_revision_number = 6

        return setting_values, variable_revision_number


class MORDObjectNameSubscriber(LabelSubscriber):
    """An object name subscriber limited by the objects in the objects' group"""

    def set_module(self, module):
        assert isinstance(module, MeasureObjectIntensityDistribution)
        self.__module = module

    def __is_valid_choice(self, choice_tuple):
        for object_group in self.__module.objects:
            if choice_tuple[0] == object_group.object_name:
                return True
        return False

    def get_choices(self, pipeline):
        super_choices = super(self.__class__, self).get_choices(pipeline)
        return list(filter(self.__is_valid_choice, super_choices))

    def is_visible(self):
        """Return True if a choice should be displayed"""
        return len(self.__module.objects) > 1

    def get_objects_name(self):
        """Return the name of the objects to use in the display"""
        if len(self.__module.objects) == 1:
            return self.__module.objects[0].object_name.value
        return self.value


class MORDImageNameSubscriber(ImageSubscriber):
    """An image name subscriber limited by the images in the image group"""

    def set_module(self, module):
        assert isinstance(module, MeasureObjectIntensityDistribution)
        self.__module = module

    def __is_valid_choice(self, choice_tuple):
        for image_name in self.__module.images_list.value:
            if choice_tuple[0] == image_name:
                return True
        return False

    def get_choices(self, pipeline):
        super_choices = super(self.__class__, self).get_choices(pipeline)

        return list(filter(self.__is_valid_choice, super_choices))

    def is_visible(self):
        """Return True if a choice should be displayed"""
        return len(self.__module.images_list.value) > 1

    def get_image_name(self):
        """Return the name of the image to use in the display"""
        if len(self.__module.images_list.value) == 1:
            return self.__module.images_list.value[0]
        return self.value
