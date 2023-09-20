How To Use The Image Tools
==========================

Right-clicking in an image displayed in a window will bring up a pop-up
menu with the following options:

-  *Open image in new window:* Displays the image in a new display
   window. This is useful for getting a closer look at a window subplot
   that has a small image.
-  *Show image histogram:* Produces a new window containing a histogram
   of the pixel intensities in the image. This is useful for
   qualitatively examining whether a threshold value determined by
   **IdentifyPrimaryObjects** seems reasonable, for example. Image
   intensities in CellProfiler typically range from zero (dark) to one
   (bright). If you have an RGB image, the histogram shows the intensity
   values for all three channels combined, even if one or more channels
   is turned off for viewing.
-  *Image contrast:* Presents three options for displaying the
   color/intensity values in the images:

      -  *Raw:* Shows the image using the full colormap range permissible for
         the image type. For example, for a 16-bit image, the pixel data will
         be shown using 0 as black and 65535 as white. However, if the actual
         pixel intensities span only a portion of the image intensity range,
         this may render the image unviewable. For example, if a 16-bit image
         only contains 12 bits of data, the resulting image will be entirely
         black.
      -  *Normalized (default):* Shows the image with the colormap
         “autoscaled” to the maximum and minimum pixel intensity values; the
         minimum value is black and the maximum value is white.
      -  *Log normalized:* Same as *Normalized* except that the color values
         are then log transformed. This is useful for when the pixel intensity
         spans a wide range of values but the standard deviation is small
         (e.g., the majority of the interesting information is located at the
         dim values). Using this option increases the effective contrast.

-  *Interpolation:* Presents three options for displaying the resolution
   in the images. This is useful for specifying the amount of detail
   that you want to be visible if you zoom in:

      -  *Nearest neighbor:* Use the intensity of the nearest image pixel when
         displaying screen pixels at sub-pixel resolution. This produces a
         blocky image, but the image accurately reflects the data.
      -  *Linear:* Use the weighted average of the four nearest image pixels
         when displaying screen pixels at sub-pixel resolution. This produces
         a smoother, more visually-appealing image, but makes it more
         difficult to find pixel borders.
      -  *Cubic:* Perform a bicubic interpolation of the nearby image pixels
         when displaying screen pixels at sub-pixel resolution. This produces
         the most visually-appealing image but is the least faithful to the
         image pixel values.

-  *Save subplot:* Save the clicked subplot as an image file. If there
   is only one plot in the figure, this option will save that one.
-  *Channels:* For color images only. You can show any combination of
   the red, green, and blue color channels.
