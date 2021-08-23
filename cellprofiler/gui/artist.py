# coding=utf-8
"""artist.py - Specialized matplotlib artists for CellProfiler
"""

import centrosome.cpmorphology
import centrosome.outline
import matplotlib
import matplotlib.artist
import matplotlib.cm
import matplotlib.collections
import matplotlib.colors
import matplotlib.image
import numpy
import scipy.ndimage
import skimage.exposure
import skimage.transform

import cellprofiler.gui.tools

"""Render the image in shades of gray"""
MODE_GRAYSCALE = "grayscale"
"""Render the image in shades of a color"""
MODE_COLORIZE = "colorize"
"""Render the image using a color map"""
MODE_COLORMAP = "colormap"
"""Render the image as RGB"""
MODE_RGB = "rgb"
"""Do not display"""
MODE_HIDE = "hide"

NORMALIZE_RAW = "raw"
NORMALIZE_LOG = "log"
NORMALIZE_LINEAR = "linear"

INTERPOLATION_NEAREST = "nearest"
INTERPOLATION_BILINEAR = "bilinear"
INTERPOLATION_BICUBIC = "bicubic"
# Map MPL interpolation modes to skimage order values
INTERPOLATION_MAP = {INTERPOLATION_NEAREST: 0, INTERPOLATION_BILINEAR: 1, INTERPOLATION_BICUBIC: 3}

MODE_OUTLINES = "outlines"
MODE_LINES = "lines"
MODE_OVERLAY = "overlay"
MODE_INVERTED = "inverted"


class ColorMixin(object):
    """A mixin for burying the color, colormap and alpha in hidden attributes
    """

    def __init__(self):
        self._alpha = 1
        self._color = None
        self._colormap = None

    def _set_alpha(self, alpha):
        self._alpha = alpha
        self._on_alpha_changed()

    def _get_alpha(self):
        return self._alpha

    def get_alpha(self):
        return self._get_alpha()

    def set_alpha(self, alpha):
        self._set_alpha(alpha)

    alpha = property(get_alpha, set_alpha)

    @staticmethod
    def _on_alpha_changed():
        """Called when the alpha value is modified, default = do nothing"""
        pass

    def _get_color(self):
        """Get the color - default is the matplotlib foreground color"""
        if self._color is None:
            color = matplotlib.rcParams.get("patch.facecolor", "b")
        else:
            color = self._color
        return numpy.atleast_1d(matplotlib.colors.colorConverter.to_rgb(color))

    def _set_color(self, color):
        """Set the color"""
        self._color = color
        self._on_color_changed()

    def get_color(self):
        return ColorMixin._get_color(self)

    def set_color(self, color):
        ColorMixin._set_color(self, color)

    color = property(get_color, set_color)

    @property
    def color3(self):
        """Return the color as a 3D array"""
        return self.color[numpy.newaxis, numpy.newaxis, :]

    def _on_color_changed(self):
        """Called when the color changed, default = do nothing"""
        pass

    def _get_colormap(self):
        """Override to get the colormap"""
        if self._colormap is None:
            return matplotlib.rcParams.get("image.cmap", "jet")
        return self._colormap

    def _set_colormap(self, colormap):
        """Override to set the colormap"""
        self._colormap = colormap
        self._on_colormap_changed()

    def get_colormap(self):
        return self._get_colormap()

    def set_colormap(self, colormap):
        self._set_colormap(colormap)

    colormap = property(get_colormap, set_colormap)

    def _on_colormap_changed(self):
        """Called when the colormap changed, default = do nothing"""
        pass

    @property
    def using_alpha(self):
        """True if this data's configuration will use the alpha setting"""
        return self._using_alpha()

    @staticmethod
    def _using_alpha():
        """Override if we ever don't use the alpha setting"""
        return True

    @property
    def using_color(self):
        """True if this data's configuration will use the color setting"""
        return self._using_color()

    def _using_color(self):
        """Override this to tell the world that the data will use the color setting"""
        return False

    @property
    def using_colormap(self):
        """True if this data's configuration will use the colormap setting"""
        return self._using_colormap()

    def _using_colormap(self):
        """Override this to tell the world that the data will use the colormap setting"""
        return False


class ImageData(ColorMixin):
    """The data and metadata needed to display an image

             name - the name of the channel
             pixel_data - a 2D or 2D * n-channel array of values
             mode - MODE_GRAYSCALE
             color - if present, render this image using values between 0,0,0
                     and this color. Default is white
             colormap - if present, use this color map for scaling
             alpha - when compositing an image on top of another, use this
                     alpha value times the color.
             normalization - one of the NORMALIZE_ constants
             vmin - the floor of the image range in raw mode - default is 0
             vmax - the ceiling of the image range in raw mode - default is 1
    """

    def __init__(
        self,
        name,
        pixel_data,
        mode=None,
        color=(1.0, 1.0, 1.0),
        colormap=None,
        alpha=None,
        normalization=None,
        vmin=0,
        vmax=1,
    ):
        super(ImageData, self).__init__()
        self.name = name
        self.pixel_data = pixel_data
        self.__mode = mode
        self.color = color
        self.colormap = colormap
        if alpha is not None:
            self.alpha = alpha
        self.normalization = normalization
        self.vmin = vmin
        self.vmax = vmax

    def set_mode(self, mode):
        self.__mode = mode

    def get_mode(self):
        if self.__mode == MODE_HIDE:
            return MODE_HIDE
        if self.pixel_data.ndim == 3:
            return MODE_RGB
        elif self.__mode is None:
            return MODE_GRAYSCALE
        else:
            return self.__mode

    def get_raw_mode(self):
        """Get the mode as set by set_mode or in the constructor"""
        return self.__mode

    mode = property(get_mode, set_mode)

    def _using_color(self):
        return self.mode == MODE_COLORIZE

    def _using_colormap(self):
        return self.mode == MODE_COLORMAP


class OutlinesMixin(ColorMixin):
    """Provides rendering of "labels" as outlines

    Needs self.labels to return a sequence of labels matrices and needs
    self._flush_outlines to be called when parameters change.
    """

    def __init__(self, outline_color, line_width):
        super(OutlinesMixin, self).__init__()
        self._flush_outlines()
        self.color = outline_color
        self._line_width = line_width

    def _flush_outlines(self):
        self._outlines = None
        self._points = None

    def _on_color_changed(self):
        if self._points is not None:
            self._points.set_color(self.color)

    def get_line_width(self):
        return self._line_width

    def set_line_width(self, line_width):
        self._line_width = line_width
        self._outlines = None
        if self._points is not None:
            self._points.set_linewidth(line_width)

    line_width = property(get_line_width, set_line_width)

    @property
    def outlines(self):
        """Get a mask of all the points on the border of objects"""
        if self._outlines is None:
            for i, labels in enumerate(self.labels):
                if i == 0:
                    self._outlines = centrosome.outline.outline(labels) != 0
                else:
                    self._outlines |= centrosome.outline.outline(labels) != 0
            if self.line_width is not None and self.line_width > 1:
                hw = float(self.line_width) / 2
                d = scipy.ndimage.distance_transform_edt(~self._outlines)
                dti, dtj = numpy.where((d < hw + 0.5) & ~self._outlines)
                self._outlines = self._outlines.astype(numpy.float32)
                self._outlines[dti, dtj] = numpy.minimum(1, hw + 0.5 - d[dti, dtj])

        return self._outlines.astype(numpy.float32)

    @property
    def points(self):
        """Return an artist for drawing the points"""
        if self._points is None:
            self._points = CPOutlineArtist(
                self.name, self.labels, linewidth=self.line_width, color=self.color
            )
        return self._points


class ObjectsData(OutlinesMixin):
    """The data needed to display objects

    name - the name of the objects
    labels - a sequence of label matrices
    outline_color - render outlines in this color. Default is
    primary color.
    line_width - the width of the line in pixels for outlines
    colormap - render overlaid labels using this colormap
    alpha - render overlaid labels using this alpha value.
    mode - the display mode: outlines, lines, overlay or hide
    scramble - True (default) to scramble the colors. False to use
                         the labels to pick from the colormap.
    """

    def __init__(
        self,
        name,
        labels,
        outline_color=None,
        line_width=None,
        colormap=None,
        alpha=None,
        mode=None,
        scramble=True,
    ):
        super(ObjectsData, self).__init__(outline_color, line_width)
        self.name = name
        self.__labels = labels
        self.colormap = colormap
        if alpha is not None:
            self.alpha = alpha
        self.mode = mode
        self.scramble = scramble
        self.__overlay = None
        self.__mask = None

    def get_labels(self):
        return self.__labels

    def set_labels(self, labels):
        self.__labels = labels
        self._flush_outlines()
        self.__overlay = None
        self.__mask = None

    def get_raw_mode(self):
        return self.mode

    labels = property(get_labels, set_labels)

    def _on_colormap_changed(self):
        super(ObjectsData, self)._on_colormap_changed()
        self.__overlay = None

    def _using_color(self):
        return self.mode in (MODE_LINES, MODE_OUTLINES)

    def _using_colormap(self):
        return self.mode == MODE_OVERLAY

    @property
    def overlay(self):
        """Return a color image of the segmentation as an overlay
        """
        if self.__overlay is not None:
            return self.__overlay
        sm = matplotlib.cm.ScalarMappable(cmap=self.colormap)
        sm.set_clim(
            vmin=1, vmax=numpy.max([numpy.max(label) for label in self.labels]) + 1
        )

        img = None
        lmin = 0
        for label in self.labels:
            if numpy.all(label == 0):
                continue
            if self.scramble:
                lmin = numpy.min(label[label != 0])
            label[label != 0] = (
                cellprofiler.gui.tools.renumber_labels_for_display(label)[label != 0]
                + lmin
            )
            lmin = numpy.max(label)
            if img is None:
                img = sm.to_rgba(label)
                img[label == 0, :] = 0
            else:
                img[label != 0, :] = sm.to_rgba(label[label != 0])
        self.__overlay = img
        return img

    @property
    def mask(self):
        """Return a mask of the labeled portion of the field of view"""
        if self.__mask is None:
            self.__mask = self.labels[0] != 0
            for l in self.labels[1:]:
                self.__mask[l != 0] = True
        return self.__mask


class MaskData(OutlinesMixin):
    """The data needed to display masks

    name - name of the mask
    mask - the binary mask
    mode - the display mode: outline, lines, overlay or inverted
    color - color of outline or line or overlay
    alpha - alpha of the outline or overlay

    MODE_OVERLAY colors the part of the mask that covers the part of the
    image that the user cares about. Ironically, the user almost certainly
    wants to see that part and MODE_INVERTED with an alpha of 1 masks the
    part that the user does not want to see and ignores the part they do.
    """

    def __init__(self, name, mask, mode=None, color=None, line_width=None, alpha=None):
        super(MaskData, self).__init__(color, line_width)
        self.name = name
        self.mask = mask
        self.mode = mode
        if alpha is not None:
            self.alpha = alpha

    @property
    def labels(self):
        """Return the mask as a sequence of labels matrices"""
        return [self.mask.astype(numpy.uint8)]

    def _using_color(self):
        return True

    def get_raw_mode(self):
        return self.mode


class CPImageArtist(matplotlib.artist.Artist):
    """An artist that displays multiple images and objects

    The image artist maintains each image and object set separately as
    well as separate rendering colors, interpolation, intensity normalization
    and display styles.

    The keyword arguments:

    images - a sequence of ImageData objects to be composited.

    objects - a sequence of ObjectsData to be composited.


    masks - a sequence of masks
            name - the name of the mask
            mask - the binary matrix for the mask
    """

    MI_IMAGES = "Images"
    MI_OBJECTS = "Objects"
    MI_MASKS = "Masks"
    MI_INTERPOLATION = "Interpolation"
    MI_NEAREST_NEIGHBOR = "Nearest neighbor"
    MI_BILINEAR = "Bilinear"
    MI_BICUBIC = "Bicubic"
    MI_RAW = "Raw"
    MI_LINEAR = "Normalized"
    MI_LOG = "Log normalized"
    MI_LINES = "Lines"
    MI_OUTLINES = "Outlines"
    MI_OVERLAY = "Overlay"
    MI_MODE = "Mode"
    MI_ALPHA = "Alpha"
    MI_COLOR = "Color"
    MI_GRAYSCALE = "Grayscale"
    MI_MASK = "Mask"
    MI_INVERTED = "Inverted mask"
    MI_COLORMAP = "Color map"
    MI_NORMALIZATION = "Intensity normalization"

    def __init__(self, images=None, objects=None, masks=None, interpolation=None):
        """Initialize the artist with the images and objects"""
        super(CPImageArtist, self).__init__()
        self.__images = images or []
        self.__objects = objects or []
        self.__masks = masks or []
        self.__interpolation = interpolation
        self.filterrad = 0  # This was 4.0 but WHY?!?!

    def set_interpolation(self, interpolation):
        self.__interpolation = interpolation

    def get_interpolation(self, rcparams=None):
        return (
            self.__interpolation
            or (rcparams or matplotlib.rcParams)["image.interpolation"]
        )

    interpolation = property(get_interpolation, set_interpolation)

    def add(self, data):
        """Add an image, objects or mask to the artist

        data - ImageData, ObjectsData or MaskData to be added
        """
        assert isinstance(data, (ImageData, ObjectsData, MaskData))
        if isinstance(data, ImageData):
            self.__images.append(data)
        elif isinstance(data, ObjectsData):
            self.__objects.append(data)
        else:
            self.__masks.append(data)

    def remove(self, data):
        """Remove an image, object or mask from the artist

        data - an ImageData, ObjectData or MaskData previously
               added (via constructor or add)
        """
        assert isinstance(data, (ImageData, ObjectsData, MaskData))
        if isinstance(data, ImageData):
            self.__images.remove(data)
        elif isinstance(data, ObjectsData):
            self.__objects.remove(data)
        else:
            self.__masks.remove(data)

    def remove_image_by_name(self, name):
        """Remove an image via the name given to it in its data"""
        for data in self.__images:
            if data.name == name:
                return self.remove(data)
        else:
            raise ValueError("Could not find image named %s" % name)

    def remove_objects_by_name(self, name):
        """Remove objects via their name given to it in its data"""
        for data in self.__objects:
            if data.name == name:
                return self.remove(data)
        else:
            raise ValueError("Could not find objects named %s" % name)

    def remove_mask_by_name(self, name):
        """Remove a mask via the name given to it in its data"""
        for data in self.__masks:
            if data.name == name:
                return self.remove(data)
        else:
            raise ValueError("Could not find mask named %s" % name)

    def get_border_count(self):
        """# of pixels needed for interpolation"""
        if self.interpolation == INTERPOLATION_NEAREST:
            return 1
        elif self.interpolation == INTERPOLATION_BICUBIC:
            return 2
        else:
            return 3

    @property
    def mp_interpolation(self):
        """Matplotlib-based interpolation constant"""
        if self.interpolation == INTERPOLATION_BICUBIC:
            return "bilinear"
        elif self.interpolation == INTERPOLATION_BILINEAR:
            return "bilinear"
        return "nearest"

    def get_channel_values(self, x, y):
        """Return a map of channel name to intensity at the given location

        x, y - coordinate location
        """
        if x < 0 or y < 0:
            return {}
        result = {}
        for image in self.__images:
            if image.mode != MODE_HIDE:
                pixel_data = image.pixel_data
                if y >= pixel_data.shape[0] or x >= pixel_data.shape[1]:
                    continue
                if pixel_data.ndim == 3:
                    value = numpy.mean(pixel_data[y, x, :])
                else:
                    value = pixel_data[y, x]
                result[image.name] = value
        return result

    def draw(self, renderer, *args, **kwargs):
        magnification = renderer.get_image_magnification()
        shape = [0, 0]
        for image in self.__images:
            if image.mode == MODE_HIDE:
                continue
            for i in range(2):
                shape[i] = max(shape[i], image.pixel_data.shape[i])
        if any([x == 0 for x in shape]):
            return
        border = self.get_border_count()

        vl = self.axes.viewLim
        view_xmin = int(max(0, min(vl.x0, vl.x1) - self.filterrad))
        view_ymin = int(max(0, min(vl.y0, vl.y1) - self.filterrad))
        view_xmax = int(min(shape[1], max(vl.x0, vl.x1) + self.filterrad))
        view_ymax = int(min(shape[0], max(vl.y0, vl.y1) + self.filterrad))
        flip_ud = vl.y0 > vl.y1
        flip_lr = vl.x0 > vl.x1
        if shape[1] <= view_xmin or shape[1] <= -view_xmin or view_xmax <= 0:
            return
        if shape[0] <= view_ymin or shape[0] <= -view_ymin or view_ymax <= 0:
            return

        # First 3 color indices are intensities
        # Last is the alpha

        target = numpy.zeros(
            (view_ymax - view_ymin, view_xmax - view_xmin, 4), numpy.float32
        )

        def get_tile_and_target(pixel_data):
            """Return the visible tile of the image and a view of the target"""
            xmin = max(0, view_xmin)
            ymin = max(0, view_ymin)
            xmax = min(view_xmax, pixel_data.shape[1])
            ymax = min(view_ymax, pixel_data.shape[0])
            if pixel_data.ndim == 3:
                pixel_data = pixel_data[ymin:ymax, xmin:xmax, :]
            else:
                pixel_data = pixel_data[ymin:ymax, xmin:xmax]
            target_view = target[: (ymax - view_ymin), : (xmax - view_xmin), :]
            return pixel_data, target_view

        max_color_in = numpy.zeros(3)
        for image in self.__images:
            assert isinstance(image, ImageData)
            if image.mode == MODE_HIDE:
                continue
            if image.pixel_data.shape[1] <= abs(view_xmin) or image.pixel_data.shape[
                0
            ] <= abs(view_ymin):
                continue
            pixel_data, target_view = get_tile_and_target(image.pixel_data)
            tv_alpha = target_view[:, :, 3]
            tv_image = target_view[:, :, :3]
            if pixel_data.dtype == "bool":
                image.normalization = NORMALIZE_RAW
            if image.normalization in (NORMALIZE_LINEAR, NORMALIZE_LOG):
                pd_max = numpy.max(pixel_data)
                pd_min = numpy.min(pixel_data)
                if pd_min == pd_max:
                    pixel_data = numpy.zeros(pixel_data.shape, numpy.float32)
                else:
                    pixel_data = (pixel_data - pd_min) / (pd_max - pd_min)
            else:
                pixel_data = pixel_data.copy()
                pixel_data[pixel_data < image.vmin] = image.vmin
                pixel_data[pixel_data > image.vmax] = image.vmax
            if image.normalization == NORMALIZE_LOG:
                log_eps = numpy.log(1.0 / 256)
                log_one_plus_eps = numpy.log(257.0 / 256)
                pixel_data = (numpy.log(pixel_data + 1.0 / 256) - log_eps) / (
                    log_one_plus_eps - log_eps
                )
            if image.mode == MODE_COLORIZE or image.mode == MODE_GRAYSCALE:
                pixel_data = pixel_data[:, :, numpy.newaxis] * image.color3
            elif image.mode == MODE_COLORMAP:
                sm = matplotlib.cm.ScalarMappable(cmap=image.colormap)
                if image.normalization == NORMALIZE_RAW:
                    sm.set_clim((image.vmin, image.vmax))
                pixel_data = sm.to_rgba(pixel_data)[:, :, :3]
            max_color_in = numpy.maximum(
                max_color_in,
                numpy.max(
                    pixel_data.reshape(
                        pixel_data.shape[0] * pixel_data.shape[1], pixel_data.shape[2]
                    ),
                    0,
                ),
            )
            imalpha = image.alpha
            tv_image[:] = (
                tv_image * tv_alpha[:, :, numpy.newaxis] * (1 - imalpha)
                + pixel_data * imalpha
            )
            tv_alpha[:] = tv_alpha + imalpha - tv_alpha * imalpha
            tv_image[tv_alpha != 0, :] /= tv_alpha[tv_alpha != 0][:, numpy.newaxis]

        #
        # Normalize the image intensity
        #
        max_color_out = numpy.max(
            target[:, :, :3].reshape(target.shape[0] * target.shape[1], 3), 0
        )
        color_mask = (max_color_in != 0) & (max_color_out != 0)
        if numpy.any(color_mask):
            multiplier = numpy.min(max_color_in[color_mask] / max_color_out[color_mask])
        else:
            multiplier = 1
        target[:, :, :3] *= multiplier

        for om in list(self.__objects) + list(self.__masks):
            assert isinstance(om, OutlinesMixin)
            if isinstance(om, MaskData) and om.mode == MODE_LINES:
                # Lines mode currently not working with Masks
                om.mode = MODE_OUTLINES
                om.alpha = 0.8
            if om.mode in (MODE_LINES, MODE_HIDE):
                continue
            if om.mode == MODE_OUTLINES:
                oshape = om.outlines.shape
                if oshape[1] <= abs(view_xmin) or oshape[0] <= abs(view_ymin):
                    continue
                mask, target_view = get_tile_and_target(om.outlines)
                oalpha = mask.astype(float) * om.alpha
                ocolor = om.color3
            elif isinstance(om, ObjectsData) and om.mode == MODE_OVERLAY:
                oshape = om.outlines.shape
                if oshape[1] <= abs(view_xmin) or oshape[0] <= abs(view_ymin):
                    continue
                ocolor, target_view = get_tile_and_target(om.overlay[:, :, :3])
                mask, _ = get_tile_and_target(om.mask)
            elif isinstance(om, MaskData) and om.mode in (MODE_OVERLAY, MODE_INVERTED):
                mask = om.mask
                if mask.shape[1] <= abs(view_xmin) or mask.shape[0] <= abs(view_ymin):
                    continue
                mask, target_view = get_tile_and_target(mask)
                if om.mode == MODE_INVERTED:
                    mask = ~mask
                ocolor = mask[:, :, numpy.newaxis] * om.color3
            else:
                continue
            tv_alpha = target_view[:, :, 3]
            tv_image = target_view[:, :, :3]
            tv_alpha3 = tv_alpha[:, :, numpy.newaxis]
            oalpha = mask.astype(float) * om.alpha
            oalpha3 = oalpha[:, :, numpy.newaxis]
            tv_image[:] = tv_image * tv_alpha3 * (1 - oalpha3) + ocolor * oalpha3
            tv_alpha[:] = tv_alpha + oalpha - tv_alpha * oalpha
            tv_image[tv_alpha != 0, :] /= tv_alpha[tv_alpha != 0][:, numpy.newaxis]

        target = target[:, :, :3]

        numpy.clip(target, 0, 1, target)

        if flip_lr:
            target = numpy.fliplr(target)

        if self.axes.viewLim.height < 0:
            target = numpy.flipud(target)

        # im = matplotlib.image.fromarray(target[:, :, :3], 0)
        # im.is_grayscale = False
        # im.set_interpolation(self.mp_interpolation)
        fc = matplotlib.rcParams["axes.facecolor"]
        bg = matplotlib.colors.colorConverter.to_rgba(fc, 0)
        # im.set_bg(*bg)

        # image input dimensions
        # im.reset_matrix()

        # the viewport translation in the X direction
        tx = view_xmin - min(vl.x0, vl.x1) - 0.5

        #
        # the viewport translation in the Y direction
        # which is from the bottom of the screen
        #
        # if self.axes.viewLim.height < 0:
        # ty = (view_ymin - self.axes.viewLim.y1) - .5
        # ty = self.axes.viewLim.y0 - view_ymax + 0.5
        # else:
        #     ty = view_ymin - self.axes.viewLim.y0 - 0.5

        # im.apply_translation(tx, ty)

        l, b, r, t = self.axes.bbox.extents

        if b > t:
            t, b = b, t

        width_display = (r - l + 1) * magnification
        height_display = (t - b + 1) * magnification

        # resize viewport to display
        sx = width_display / self.axes.viewLim.width
        sy = abs(height_display / self.axes.viewLim.height)

        # im.apply_scaling(sx, sy)
        # im.resize(width_display, height_display, norm=1, radius=self.filterrad)

        bounding_box = self.axes.bbox.frozen()

        graphics_context = renderer.new_gc()

        graphics_context.set_clip_rectangle(bounding_box)

        image = numpy.zeros((target.shape[0], target.shape[1], 4), numpy.uint8)

        image[:, :, 3] = 255

        image[:, :, :3] = skimage.exposure.rescale_intensity(
            target, out_range=numpy.uint8
        )

        image = skimage.transform.rescale(image, (sx, sy, 1), order=INTERPOLATION_MAP[self.mp_interpolation])

        image = skimage.img_as_ubyte(image)

        im_xmin = int(min(vl.x0, vl.x1))
        im_xmax = int(max(vl.x0, vl.x1))
        im_ymin = int(min(vl.y0, vl.y1))
        im_ymax = int(max(vl.y0, vl.y1))
        # Correct drawing start point when origin is not 0
        if im_xmin < 0:
            l = ((0 - im_xmin) / (im_xmax - im_xmin) * (r - l)) + l
        if im_ymax > shape[0]:  # origin corresponds to max y, not 0:
            b = ((im_ymax - shape[0]) / (im_ymax - im_ymin) * (t - b)) + b

        renderer.draw_image(graphics_context, l, b, image)

        for om in list(self.__objects) + list(self.__masks):
            assert isinstance(om, OutlinesMixin)
            if om.mode == MODE_LINES:
                om.points.axes = self.axes
                om.points.set_transform(self.axes.transData)
                om.points.set_clip_path(self.axes.patch)
                om.points.draw(renderer)

    def add_to_menu(self, target, menu):
        """Add to a context menu for a WX ui

        target - target window that will receive menu events.
        """
        import wx

        assert isinstance(menu, wx.Menu)
        interpolation_menu = wx.Menu()
        assert isinstance(menu, wx.Menu)
        menu.AppendSeparator()
        menu.AppendSubMenu(interpolation_menu, self.MI_INTERPOLATION)
        for label, state in (
            (self.MI_NEAREST_NEIGHBOR, INTERPOLATION_NEAREST),
            (self.MI_BILINEAR, INTERPOLATION_BILINEAR),
            (self.MI_BICUBIC, INTERPOLATION_BICUBIC),
        ):
            my_id = wx.NewId()
            submenu_item = interpolation_menu.AppendRadioItem(my_id, label)
            target.Bind(
                wx.EVT_MENU,
                (
                    lambda event, target=state: self.on_interpolation_menu_event(
                        event, target
                    )
                ),
                id=my_id,
            )
            target.Bind(
                wx.EVT_UPDATE_UI,
                (
                    lambda event, target=state: self.on_interpolation_update_event(
                        event, target
                    )
                ),
                id=my_id,
            )
            if state == self.interpolation:
                submenu_item.Check(True)
        menu.AppendSeparator()

    def on_interpolation_menu_event(self, event, target):
        self.interpolation = target
        self.refresh()

    def on_interpolation_update_event(self, event, target):
        import wx

        assert isinstance(event, wx.UpdateUIEvent)
        event.Check(self.interpolation == target)

    def on_update_menu(self, event, menu):
        import wx

        assert isinstance(menu, wx.Menu)
        menu_items = list(menu.GetMenuItems())
        breaks = (
            (self.MI_IMAGES, self.__images),
            (self.MI_OBJECTS, self.__objects),
            (self.MI_MASKS, self.__masks),
        )
        for start, item in enumerate(menu_items):
            assert isinstance(item, wx.MenuItem)
            if item.ItemLabel == self.MI_INTERPOLATION:
                break
        else:
            return
        window = self.__get_window_from_event(event)
        idx = start + 1
        if menu_items[idx].IsSeparator():
            idx += 1
        label_fmt = "--- %s ---"
        for key, sequence in breaks:
            label = label_fmt % key
            if idx >= len(menu_items) or menu_items[idx].ItemLabel != label:
                item = menu.Insert(idx, wx.NewId(), label)
                item.Enable(False)
                menu_items.insert(idx, item)
            idx += 1
            #
            # Pair data items with menu items
            #
            for data in sequence:
                name = data.name
                if (
                    idx == len(menu_items)
                    or menu_items[idx].ItemLabel.startswith("---")
                    or menu_items[idx].IsSeparator()
                ):
                    sub_menu = wx.Menu()
                    my_id = wx.NewId()
                    if len(name) == 0:
                        # otherwise bad things happen on Mac
                        # Can't have blank name and non-stock ID
                        name = " "
                    sub_menu_item = menu.Insert(idx, my_id, name, sub_menu)
                    if data.mode == MODE_HIDE:
                        sub_menu_item.Enable(False)
                    menu_items.insert(idx, sub_menu_item)
                    self.__initialize_sub_menu(event, sub_menu, data)

                    def on_update_ui(event, sub_menu=sub_menu, data=data):
                        self.__update_sub_menu(event, sub_menu, data)

                    window.Bind(wx.EVT_UPDATE_UI, on_update_ui, id=my_id)
                    idx += 1
                else:
                    self.__update_sub_menu(
                        menu_items[idx], menu_items[idx].GetMenu(), data
                    )
                    idx += 1
            #
            # Remove excess menu items
            #
            while len(menu_items) < idx and menu_items[idx].IsEnabled():
                menu.Remove(item)
                del menu_items[idx]

    def __initialize_sub_menu(self, event, sub_menu, data):
        import wx

        assert isinstance(sub_menu, wx.Menu)
        if isinstance(data, ImageData):
            self.__initialize_image_sub_menu(event, sub_menu, data)
        elif isinstance(data, ObjectsData):
            self.__initialize_objects_sub_menu(event, sub_menu, data)
        elif isinstance(data, MaskData):
            self.__initialize_mask_sub_menu(event, sub_menu, data)

    def __initialize_image_sub_menu(self, event, sub_menu, data):
        import wx

        item = sub_menu.Append(wx.NewId(), self.MI_NORMALIZATION)
        item.Enable(False)
        window = self.__get_window_from_event(event)
        for label, target in (
            (self.MI_RAW, NORMALIZE_RAW),
            (self.MI_LINEAR, NORMALIZE_LINEAR),
            (self.MI_LOG, NORMALIZE_LOG),
        ):
            my_id = wx.NewId()
            sub_menu.AppendRadioItem(my_id, label)
            window.Bind(
                wx.EVT_MENU,
                (
                    lambda event, data=data, target=target: self.__on_set_normalization(
                        data, target
                    )
                ),
                id=my_id,
            )
            window.Bind(
                wx.EVT_UPDATE_UI,
                (
                    lambda event, data=data, target=target: self.__on_update_normalization(
                        event, data, target
                    )
                ),
                id=my_id,
            )
        sub_menu.AppendSeparator()
        my_id = wx.NewId()
        item = sub_menu.Append(my_id, self.MI_MODE)
        item.Enable(False)
        for label, target in (
            (self.MI_COLOR, MODE_COLORIZE),
            (self.MI_GRAYSCALE, MODE_GRAYSCALE),
            (self.MI_COLORMAP, MODE_COLORMAP),
        ):

            def update_mode(event_or_item, data=data, target=target):
                if data.mode == MODE_RGB:
                    event_or_item.Enable(False)
                else:
                    event_or_item.Enable(True)
                    event_or_item.Check(data.mode == target)

            def on_mode(event, data=data, target=target):
                data.mode = target
                self.refresh()

            my_id = wx.NewId()
            item = sub_menu.AppendRadioItem(my_id, label)
            update_mode(item)
            window.Bind(wx.EVT_MENU, on_mode, id=my_id)
            window.Bind(wx.EVT_UPDATE_UI, update_mode, id=my_id)
        sub_menu.AppendSeparator()
        self.__add_color_item(
            event, sub_menu, data, "Set image color", "Set image colormap"
        )
        self.__add_alpha_item(event, sub_menu, data, "Set image transparency")

    def __on_set_normalization(self, data, target):
        assert isinstance(data, ImageData)
        data.normalization = target
        self.refresh()

    def __add_color_item(self, event, sub_menu, data, color_msg, colormap_msg):
        import wx

        assert isinstance(data, ColorMixin)
        my_id = wx.NewId()
        item = sub_menu.Append(my_id, self.MI_COLOR)
        window = self.__get_window_from_event(event)

        def on_color(event, data=data, color_msg=color_msg, colormap_msg=colormap_msg):
            if data.using_color:
                self.__on_color_dlg(event, color_msg, data)
            elif data.using_colormap:
                self.__on_colormap_dlg(event, colormap_msg, data)

        def on_update(event_or_item, data=data):
            assert isinstance(data, ColorMixin)
            event_or_item.Enable(data.using_color or data.using_colormap)

        window.Bind(wx.EVT_MENU, on_color, id=my_id)
        window.Bind(wx.EVT_UPDATE_UI, on_update, id=my_id)
        on_update(item)

    def __add_alpha_item(self, event, sub_menu, data, msg):
        import wx

        my_id = wx.NewId()
        item = sub_menu.Append(my_id, self.MI_ALPHA)
        window = self.__get_window_from_event(event)

        def set_alpha(alpha, data=data):
            data.alpha = alpha
            self.refresh()

        def on_alpha(event, data=data, msg=msg):
            self.__on_alpha_dlg(event, msg, data)

        def update_alpha(event_or_item, data=data):
            assert isinstance(data, ColorMixin)
            event_or_item.Enable(data.using_alpha)

        window.Bind(wx.EVT_MENU, on_alpha, id=my_id)
        window.Bind(wx.EVT_UPDATE_UI, update_alpha, id=my_id)
        update_alpha(item)

    def refresh(self):
        if self.figure is not None and self.figure.canvas is not None:
            self.figure.canvas.draw_idle()

    @staticmethod
    def __get_window_from_event(event):
        import wx

        o = event.EventObject
        if isinstance(o, wx.Menu):
            return o.GetInvokingWindow()
        return o

    @staticmethod
    def __on_update_normalization(event, data, target):
        assert isinstance(data, ImageData)
        event.Check(data.normalization == target)

    @staticmethod
    def __on_update_image_color_item(event, data):
        event.Enable(data.mode != MODE_RGB)

    def __initialize_objects_sub_menu(self, event, sub_menu, data):
        import wx

        assert isinstance(data, ObjectsData)
        assert isinstance(sub_menu, wx.Menu)
        item = sub_menu.Append(wx.NewId(), "Display mode")
        item.Enable(False)
        window = self.__get_window_from_event(event)
        for label, mode in (
            (self.MI_LINES, MODE_LINES),
            (self.MI_OUTLINES, MODE_OUTLINES),
            (self.MI_OVERLAY, MODE_OVERLAY),
        ):
            my_id = wx.NewId()
            sub_menu.AppendRadioItem(my_id, label)
            window.Bind(
                wx.EVT_MENU,
                (
                    lambda event, data=data, mode=mode: self.__on_set_objects_mode(
                        event, data, mode
                    )
                ),
                id=my_id,
            )
            window.Bind(
                wx.EVT_UPDATE_UI,
                (
                    lambda event, data=data, mode=mode: self.__on_update_objects_mode(
                        event, data, mode
                    )
                ),
                id=my_id,
            )
        sub_menu.AppendSeparator()
        self.__add_color_item(
            event, sub_menu, data, "Set objects color", "Set objects colormap"
        )
        self.__add_alpha_item(event, sub_menu, data, "Set objects' transparency")

    def __on_set_objects_mode(self, event, data, mode):
        data.mode = mode
        self.refresh()

    @staticmethod
    def __on_update_objects_mode(event, data, mode):
        event.Check(mode == data.mode)

    def __initialize_mask_sub_menu(self, event, sub_menu, data):
        import wx

        assert isinstance(data, MaskData)
        assert isinstance(sub_menu, wx.Menu)
        item = sub_menu.Append(wx.NewId(), self.MI_MODE)
        item.Enable(False)
        window = self.__get_window_from_event(event)
        for label, target in (
            (self.MI_LINES, MODE_LINES),
            (self.MI_OUTLINES, MODE_OUTLINES),
            (self.MI_OVERLAY, MODE_OVERLAY),
            (self.MI_INVERTED, MODE_INVERTED),
        ):
            set_fn = lambda event, data=data, mode=target: self.__on_mask_mode(
                event, data, mode
            )
            my_id = wx.NewId()
            item = sub_menu.AppendRadioItem(my_id, label)
            self.__on_update_mask_mode(item, data, target)
            window.Bind(wx.EVT_MENU, set_fn, id=my_id)
            window.Bind(
                wx.EVT_UPDATE_UI,
                (
                    lambda event, data=data, target=target: self.__on_update_mask_mode(
                        event, data, target
                    )
                ),
                id=my_id,
            )
        sub_menu.AppendSeparator()
        self.__add_color_item(event, sub_menu, data, "Set mask color", None)
        self.__add_alpha_item(event, sub_menu, data, "Set mask transparency")

    def __on_mask_mode(self, event, data, mode):
        data.mode = mode
        self.refresh()

    @staticmethod
    def __on_update_mask_mode(event_or_item, data, target):
        """Update the menu item or UpdateUIEvent's check status

        event_or_item - either an UpdateUIEvent or MenuItem or other
              thing that has a Check method
        data - a MaskData whose mode will be checked
        target - the target state

        Either checks or unchecks the item or event, depending on whether
        the data and target matches.
        """
        event_or_item.Check(target == data.mode)

    def __on_color_dlg(self, event, msg, data):
        import wx

        assert isinstance(data, ColorMixin)
        color_data = wx.ColourData()
        orig_color = data.color
        r, g, b = [int(x * 255) for x in data.color]
        color_data.SetColour(wx.Colour(r, g, b))
        window = self.__get_window_from_event(event)
        with wx.ColourDialog(window, color_data) as dlg:
            assert isinstance(dlg, wx.ColourDialog)
            dlg.Title = msg
            if dlg.ShowModal() == wx.ID_OK:
                color_data = dlg.GetColourData()
                data.color = tuple([float(x) / 255 for x in color_data.Colour])
                self.refresh()

    def __on_colormap_dlg(self, event, msg, data):
        import wx

        assert isinstance(data, ColorMixin)
        old_colormap = data.colormap
        window = self.__get_window_from_event(event)
        with wx.Dialog(window) as dlg:
            assert isinstance(dlg, wx.Dialog)
            dlg.Title = msg
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            choices = sorted([x for x in matplotlib.cm.datad if not x.endswith("_r")])
            choice = wx.Choice(dlg, choices=choices)
            choice.SetStringSelection(old_colormap)
            dlg.Sizer.Add(choice, 0, wx.EXPAND | wx.ALL, 10)
            button_sizer = wx.StdDialogButtonSizer()
            dlg.Sizer.Add(button_sizer, 0, wx.EXPAND)
            button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
            button_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
            button_sizer.Realize()

            def on_choice(event, data=data):
                data.colormap = choice.GetStringSelection()
                self.refresh()

            choice.Bind(wx.EVT_CHOICE, on_choice)
            dlg.Fit()
            if dlg.ShowModal() != wx.ID_OK:
                data.colormap = old_colormap
                self.refresh()

    def __on_alpha_dlg(self, event, msg, data):
        import wx

        assert isinstance(data, ColorMixin)
        old_alpha = data.alpha
        window = self.__get_window_from_event(event)
        with wx.Dialog(window) as dlg:
            assert isinstance(dlg, wx.Dialog)
            dlg.Title = msg
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            slider = wx.Slider(
                dlg,
                value=int(old_alpha * 255),
                minValue=0,
                maxValue=255,
                style=wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_LABELS,
            )
            slider.SetMinSize((180, slider.GetMinHeight()))
            dlg.Sizer.Add(slider, 0, wx.EXPAND | wx.ALL, 10)
            button_sizer = wx.StdDialogButtonSizer()
            dlg.Sizer.Add(button_sizer, 0, wx.EXPAND)
            button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
            button_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
            button_sizer.Realize()

            def on_slider(event, data=data):
                data.alpha = float(slider.GetValue()) / 255
                self.refresh()

            slider.Bind(wx.EVT_SLIDER, on_slider)
            dlg.Fit()
            if dlg.ShowModal() != wx.ID_OK:
                data = old_alpha
                self.refresh()

    @staticmethod
    def __update_sub_menu(event, sub_menu, data):
        event.Enable(data.mode != MODE_HIDE)
        event.Text = data.name if len(data.name) > 0 else " "


class CPOutlineArtist(matplotlib.collections.LineCollection):
    """An artist that is a plot of the outline around an object

    This class is here so that we can add and remove artists for certain
    outlines.
    """

    def set_paths(self):
        pass

    def __init__(self, name, labels, *args, **kwargs):
        """Draw outlines for objects

        name - the name of the outline

        labels - a sequence of labels matrices

        kwargs - further arguments for Line2D
        """
        # get_outline_pts has its faults:
        # * it doesn't do holes
        # * it only does one of two disconnected objects
        #
        # We get around the second failing by resegmenting with
        # connected components and combining the original and new segmentation
        #
        lines = []
        for l in labels:
            new_labels, counts = scipy.ndimage.label(l != 0, numpy.ones((3, 3), bool))
            if counts == 0:
                continue
            l = l.astype(numpy.uint64) * counts + new_labels
            unique, idx = numpy.unique(l.flatten(), return_inverse=True)
            if unique[0] == 0:
                my_range = numpy.arange(len(unique))
            else:
                my_range = numpy.arange(1, len(unique))
            idx.shape = l.shape
            pts, offs, counts = centrosome.cpmorphology.get_outline_pts(idx, my_range)
            pts = pts + 0.5  # Target the centers of the pixels.
            pts = pts[:, ::-1]  # matplotlib x, y reversed from i,j
            for off, count in zip(offs, counts):
                lines.append(numpy.vstack((pts[off : off + count], pts[off : off + 1])))
        matplotlib.collections.LineCollection.__init__(self, lines, *args, **kwargs)

    def get_outline_name(self):
        return self.__outline_name
