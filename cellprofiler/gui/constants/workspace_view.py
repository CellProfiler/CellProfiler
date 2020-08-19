from cellprofiler_core.preferences import INTENSITY_MODE_HELP, INTERPOLATION_MODE_HELP

C_CHOOSER = 0
C_COLOR = 1
C_SHOW = 2
C_REMOVE = 3

WORKSPACE_VIEWER_HELP = f"""
The workspace viewer is a flexible tool that you can use to explore your
images, objects and measurements in test mode. To use the viewer, select
*View Workspace* from the *Test* menu after starting test mode. This
will display the CellProfiler Workspace, a window with an image pane to
the left and a panel of controls to the right.

Key concepts
~~~~~~~~~~~~

The workspace viewer lets you examine the CellProfiler workspace as you
progress through your pipeline's execution. At the start of the
pipeline, the only things that are available are the images and objects
loaded by the input modules. New images, objects and measurements are
added to the workspace as you step through modules and, if you modify a
module's setting and re-execute the module, the images, objects and
measurements produced by that module will be overwritten.

The viewer is persistent. You can set up the viewer to view the
workspace at the end of a pipeline and then start a new pipeline cycle
and CellProfiler will fill in the images, objects and measurements that
you have chosen to display as they become available. You can also zoom
in on a particular region and change settings and the viewer will remain
focused on that region.

All elements of the display are configurable, either through the context
menu or through the subplots menu.

Workspace configuration parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The interpolation mode used to render images, objects and masks is a
configuration option that applies to the entire workspace.
Interpolation controls how the intensities of pixels are blended
together. You can set the interpolation mode by selecting
*Interpolation* from the *Subplots* menu. The available modes are:

{INTERPOLATION_MODE_HELP}


Images
~~~~~~

The workspace viewer can display any image that is available from the
input modules or from modules previously executed. To display a single
image, select it from the *Images* drop down and check the "Show"
checkbox. Initially, the image will be displayed in color, using the
color shown in the "Color" box. This color can be changed by clicking
on the color box.

You can add images to the display by clicking the *Add image* button.
You can remove images other than the first by hitting the button in
the *Remove* column. You can toggle image display using the checkbox
in the *Show* column.

You can change the way an image is scaled, you can change its display
mode and you can change its color and transparency from the menus. To
do this, select the image from the *Subplots* menu. The images that
are shown will appear in the menu under the *--- Images ---* heading.
Select the image you want to configure from the menu to display the
options that are available for that image. There are three categories
in the menu, one for intensity normalization, one for the display mode
and one to adjust color and transparency.

The intensity normalization mode controls how the pixel's intensity
value is translated into the brightness of a pixel on the screen. The
modes that are available are:

{INTENSITY_MODE_HELP}

The *Mode* controls how pixel intensities are mapped to colors in the image.
You can display each image using the following modes:

-  *Color:* Pixels will have a uniform color which can be selected by
   either clicking on the *Color* button next to the image name or by
   choosing the image's *Color* menu entry.
-  *Grayscale:* The image will be rendered in shades of gray. The color
   choice will have no effect and the image's *Color* menu entry will be
   unavailable.
-  *Color map:* The image will be rendered using a palette. Your default
   color map will be used initially. To change the color map, select the
   image's *Color* menu entry from its menu and choose one of the color
   maps from the drop-down. The display will change interactively as you
   change the selection, allowing you to see the image as rendered by
   your choice. Hit *OK* to accept the new color map or hit *Cancel* to
   use the color map that was originally selected.

The image's *Alpha* menu entry lets you control the image's
transparency. This will let you blend colors when the palettes overlap
and choose which image's intensity has the highest priority. To change
the transparency, select *Alpha* from the image's menu. Adjust the
transparency interactively using the slider bar and hit *OK* to accept
the new value or *Cancel* to restore the value that was originally
selected.

Objects
~~~~~~~

You can display the objects that have been created or loaded by all
modules that have been executed. To display a set of objects, select
them from the *Objects* drop-down and check the *Show* checkbox. You can
add additional objects by pressing the *Add objects* button. You can
configure the appearance of objects using the context or *Subplots*
menu. Choose the objects you wish to configure from the *--- Objects
---* list in the menu. You will see configuration menu items for the
objects' display mode, color and alpha value. You can display objects
using one of the following modes:

*Lines:* This mode draws a line through the center of each pixel that
borders the background of the object or another object. It does not
display holes in the object. The line is drawn using the color shown in
the *Color* button next to the objects' name. This option does not
obscure the border pixels, but can take longer to render, especially if
there are a large number of objects.

*Outlines:* This mode displays each pixel in the object's border using
the color shown in the *Color* button next to the objects' name. This
option will display holes in unfilled objects, but the display obscures
the image underneath the border pixels.

*Overlay:* This mode displays a different color overlay over each
object's pixels. Each object is assigned a color using the default color
map initially. You can choose the color map by selecting *Color* from
the objects' menu and choosing one of the available color maps. You can
change the transparency of the overlay by choosing *Alpha* from the
objects' menu.

Masks
~~~~~

You can display the mask for any image produced by any of the modules
that have been executed. Most images are not masked. In these cases,
you can display the mask, but the display will show that the whole
image is unmasked. You can mask an image with the **MaskImage** or
**Crop** modules.

To display the mask of an image, select it from the *Masks* dropdown
and check the *Show* checkbox. The options for masks are the same as
for objects with one addition. You can invert and overlay the mask by
choosing *Inverted* from the mask's menu and the masked portion will
be displayed in color.

Measurements
~~~~~~~~~~~~

You can display any measurement produced by any of the modules that
have been executed. Image measurements will be displayed in the title
bar above the image. Object measurements will be displayed centered
over the measurement's object. To display a measurement, select it
from the *Measurements* drop-down and check the *Show* checkbox next
to the measurement. You can add a measurement by pressing the *Add
Measurement* button or remove it by checking the button in the
*Remove* column.

You can configure the font used to display an object measurement, the
color of the text, and the color, transparency and shape of the
background behind the text. To configure the measurement's appearance,
press the *Font* button to the right of the measurement. Press the
*Font* button in the *Measurement appearance* dialog to choose the
font and its size, press the *Text color* and *Background color* to
change the color used to display the text and background. Use the
*Alpha* slider to control the transparency of the background behind
the measurement text. The *Box shape* drop-down controls the shape of
the background box. The *Precision* control determines the number of
digits displayed to the right of the decimal point.
"""
