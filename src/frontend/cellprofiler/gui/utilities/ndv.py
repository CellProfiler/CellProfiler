import logging
from ndv import ArrayViewer
from ndv.models import DataWrapper, ChannelMode
from cmap import Colormap
import numpy
import dask.array.core as da
from typing import Any, Hashable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

def get_data_wrapper(img):
    if img.ndim == 2:
        LI = {
            #'Z' : None,
            'Y' : 0,
            'X' : 1,
            #'C' : None,
        }
    elif img.ndim == 3:
        LI = {
            #'Z' : None,
            'Y' : 0,
            'X' : 1,
            'C' : 2,
        }
    elif img.ndim == 4:
        LI = {
            'Z' : 0,
            'Y' : 1,
            'X' : 2,
            'C' : 3,
        }
    else:
        raise ValidationError(f"Unsupported image dimensions: {img.ndim}")

    class CustomWrapper(DataWrapper):
        PRIORITY = 10

        _li = LI

        @classmethod
        def supports(cls, obj: Any):
            return isinstance(obj, numpy.ndarray)

        @property
        def dims(self) -> tuple[Hashable, ...]:
            return tuple(self._li.keys())

        @property
        def coords(self) -> Mapping[Hashable, Sequence]:
            return {label: range(self._data.shape[idx]) for label, idx in self._li.items()}

    class CustomDaskWrapper(DataWrapper):
        PRIORITY = 10

        _li = LI

        @classmethod
        def supports(cls, obj: Any):
            if isinstance(obj, da.Array):
                return True
            return False

        def _asarray(self, data: da.Array) -> numpy.ndarray:
            return numpy.asarray(data.compute())

        def save_as_zarr(self, path: str) -> None:
            self._data.to_zarr(url=path)

        @property
        def dims(self) -> tuple[Hashable, ...]:
            return tuple(self._li.keys())

        @property
        def coords(self) -> Mapping[Hashable, Sequence]:
            return {label: range(self._data.shape[idx]) for label, idx in self._li.items()}

    if CustomDaskWrapper.supports(img):
        return CustomDaskWrapper
    elif CustomWrapper.supports(img):
        return CustomWrapper
    else:
        raise ValidationError(f"Unsupported image data type {type(img)}")

STANDARD_LUTS = [
    {'visible': True, 'cmap': Colormap('red')},
    {'visible': True, 'cmap': Colormap('green')},
    {'visible': True, 'cmap': Colormap('blue')},
]

def ndv_display(img, ndv_viewer=None):
    if ndv_viewer is None:
        LOGGER.debug("Initializing ndv")

        data_wrapper = get_data_wrapper(img)

        num_visible_axes = min(img.shape[data_wrapper._li['C']], len(STANDARD_LUTS))
        visible_axes = list(range(num_visible_axes))
        luts = {ax: STANDARD_LUTS[ax] for ax in visible_axes}
        for ax in range(num_visible_axes, img.shape[data_wrapper._li['C']]):
            luts[ax] = {'visible': False}

        ndv_viewer = ArrayViewer(
            data_wrapper(img),
            visible_axes=('Y', 'X'),
            channel_axis='C',
            channel_mode=ChannelMode.COMPOSITE,
            default_lut={'visible': False, 'cmap': Colormap('viridis')},
            luts=luts
        )

        # TODO: ndv - temporary
        #ndv_viewer._async = False
    else:
        LOGGER.debug("Updating ndv data")
        ndv_viewer.data = img

    LOGGER.debug("Rendering image for display in ndv")
    ndv_viewer.show()

    return ndv_viewer
