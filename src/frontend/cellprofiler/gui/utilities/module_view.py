import logging
import threading
import time

import wx
from cellprofiler_core.setting import ValidationError
import cellprofiler.gui.constants.module_view as mv_constants
from cellprofiler.gui.module_view._validation_request_controller import (
    ValidationRequestController,
)

LOGGER = logging.getLogger(__name__)


def text_control_name(v):
    """Return the name of a setting's text control
    v - the setting
    The text control name is built using the setting's key
    """
    return "%s_text" % (str(v.key()))


def button_control_name(v, idx=None):
    """Return the name of a setting's button

    v - the setting

    idx - if present, the index of one of several buttons for the setting
    """
    if idx is None:
        return "%s_button" % (str(v.key()))
    else:
        return "%s_button_%d" % (str(v.key()), idx)


def edit_control_name(v):
    """Return the name of a setting's edit control
    v - the setting
    The edit control name is built using the setting's key
    """
    return str(v.key())


def min_control_name(v):
    """For a range, return the control that sets the minimum value
    v - the setting
    """
    return "%s_min" % (str(v.key()))


def max_control_name(v):
    """For a range, return the control that sets the maximum value
    v - the setting
    """
    return "%s_max" % (str(v.key()))


def absrel_control_name(v):
    """For a range, return the control that chooses between absolute and relative

    v - the setting
    Absolute - far coordinate is an absolute value
    From edge - far coordinate is a distance from the far edge
    """
    return "%s_absrel" % (str(v.key()))


def x_control_name(v):
    """For coordinates, return the control that sets the x value
    v - the setting
    """
    return "%s_x" % (str(v.key()))


def y_control_name(v):
    """For coordinates, return the control that sets the y value
    v - the setting
    """
    return "%s_y" % (str(v.key()))


def category_control_name(v):
    """For measurements, return the control that sets the measurement category

    v - the setting
    """
    return "%s_category" % (str(v.key()))


def category_text_control_name(v):
    return "%s_category_text" % (str(v.key()))


def feature_control_name(v):
    """For measurements, return the control that sets the feature name

    v - the setting
    """
    return "%s_feature" % (str(v.key()))


def feature_text_control_name(v):
    return "%s_feature_text" % (str(v.key()))


def image_control_name(v):
    """For measurements, return the control that sets the image name

    v - the setting
    """
    return "%s_image" % (str(v.key()))


def image_text_control_name(v):
    return "%s_image_text" % (str(v.key()))


def object_control_name(v):
    """For measurements, return the control that sets the object name

    v - the setting
    """
    return "%s_object" % (str(v.key()))


def object_text_control_name(v):
    return "%s_object_text" % (str(v.key()))


def scale_control_name(v):
    """For measurements, return the control that sets the measurement scale

    v - the setting
    """
    return "%s_scale" % (str(v.key()))


def scale_text_ctrl_name(v):
    return "%s_scale_text" % (str(v.key()))


def combobox_ctrl_name(v):
    return "%s_combobox" % (str(v.key()))


def colorbar_ctrl_name(v):
    return "%s_colorbar" % (str(v.key()))


def help_ctrl_name(v):
    return "%s_help" % str(v.key())


def subedit_control_name(v):
    return "%s_subedit" % str(v.key())


def grid_control_name(v):
    return "%s_grid" % str(v.key())


def custom_label_name(v):
    return "%s_customlabel" % str(v.key())


def folder_label_name(v):
    return "%s_folderlabel" % str(v.key())


def encode_label(text):
    """Encode text escapes for the static control and button labels

    The ampersand (&) needs to be encoded as && for wx.StaticText
    and wx.Button in order to keep it from signifying an accelerator.
    """
    return text.replace("&", "&&")


def validate_module(pipeline, module_num, callback):
    """Validate a module and execute the callback on error on the main thread

    pipeline - a pipeline to be validated
    module_num - the module number of the module to be validated
    callback - a callback with the signature, "fn(setting, message, pipeline_data)"
    where setting is the setting that is in error and message is the message to
    display.
    """
    modules = [m for m in pipeline.modules() if m.module_num == module_num]
    if len(modules) != 1:
        return
    module = modules[0]
    level = logging.INFO
    setting_idx = None
    message = None
    try:
        level = logging.ERROR
        module.test_valid(pipeline)  # this method validates each visible
        # setting first, then the module itself.
        level = logging.WARNING
        module.test_module_warnings(pipeline)
        level = logging.INFO
    except ValidationError as instance:
        message = instance.message
        setting_idx = [m.key() for m in module.visible_settings()].index(
            instance.get_setting().key()
        )
    except Exception as e:
        LOGGER.error("Error in validation thread", e)
    wx.CallAfter(callback, setting_idx, message, level)


def validation_queue_handler():
    try:
        while mv_constants.validation_queue_keep_running:
            request = mv_constants.validation_queue.get()
            if (
                not isinstance(request, ValidationRequestController)
                or request.cancelled
            ):
                continue
            start = time.perf_counter()
            try:
                validate_module(
                    request.pipeline, request.module_num, request.callback,
                )
            except:
                pass
            # Make sure this thread utilizes less than 1/2 of GIL clock
            wait_for = max(0.25, time.perf_counter() - start)
            time.sleep(wait_for)
    except:
        LOGGER.warning("Error in validation thread.", exc_info=True)
    LOGGER.info("Exiting the pipeline validation thread")


def request_module_validation(validation_request):
    """Request that a module be validated

    """
    if mv_constants.pipeline_queue_thread is None:
        mv_constants.pipeline_queue_thread = threading.Thread(
            target=validation_queue_handler,
            name="Pipeline validation thread",
            daemon=True
        )
        mv_constants.pipeline_queue_thread.start()
    mv_constants.validation_queue.put(validation_request)


def stop_validation_queue_thread():
    """Stop the thread that handles module validation"""

    if mv_constants.pipeline_queue_thread is not None:
        mv_constants.validation_queue_keep_running = False
        mv_constants.validation_queue.put(None)
        mv_constants.pipeline_queue_thread.join()
