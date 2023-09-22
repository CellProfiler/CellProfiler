How to replicate the Identify modules on volumetric images
==========================================================

Checkout the `3D-monolayer tutorial`_ for a step by step 3D pipeline example.

**Prepare** images by applying some/all of the these modules:

- *Resize* module: Processing 3D images requires much more computation time than 2D images.
  Often, downsampling an image can yield large performance gains and at the same time
  smoothen an image to remove noise. If the objects of interest are relatively large
  compared to the pixel size, then segmentation results will minimally affect the final
  segmentation. The resized segmented objects can be scaled back up to be applied to the
  original image. Start from 0.5.

- Filters: *Gaussian* or *Median Filter* modules could homogenize the signal within the
  object of interest and reduce noise in the background. Removing such noise may assist
  object detection.

**Segmentation**, depending on the image, some of the following modules couple be applied:

To mimic *IdentifyPrimaryObjects*:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- *Threshold* module. This will separate the foreground from the background.
  The following modules could help reduce the threshold noise:

  - *Opening*: This module removes salt noise (small bright spots) in an image.
  - *DilateImage*: Expands bright shapes in an image.

- *ErodeImage*: Shrinks bright shapes in an image.

- *Closing*: Remove pepper noise (small dark spots) and connect small bright cracks.

- *RemoveHoles*: This module implements an algorithm that will remove small holes
  within the object of interest. Any remaining holes will contribute to over-segmentation
  of the object.

- *Watershed*: This module implements the watershed algorithm, which will segment
  the primary objects. Choose the "Distance" method and an appropriate Footprint size
  to scan the image for local maxima. Downsampling could result in large performance
  gains. However, it can result in blockier segmentation. For more information on the
  watershed algorithm refer to this `MATLAB blog post`_.

- *ResizeObjects* to return the segmented primary object to the size of the original
  image. If 0.5 was chosen in the Resize module, the object must be scaled up by 2.

To mimic IdentifySecondaryObjects:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After segmenting the primary objects, these objects can be used as markers to help
identify a larger secondary object surrounding them. For this, the primary objects need
to be transformed into markers from which we’ll ‘grow’ the secondary objects. The
following sequence of modules can be used to help in this step:

- *ErodeObjects*: Shrinks objects in an image by removing pixels from their edges. Here
  it could shrink the first identified object to make them more seed-like. This can be
  helpful if your primary objects are not much smaller than the secondary objects.

- *ConvertObjectsToImage*: Convert the output of Watershed module using the uint16
  format. This assigns a marker number to each object.

- *Threshold*: To threshold the second channel to be used for secondary object
  detection. Use this to identify regions which should belong to a secondary object.

- *Watershed*: The input image would be channel for the second object segmentation.
  Choose "Markers" option in the Generate from dropdown menu. The Markers will be the
  Seeds/first segmentation image converted to *uint16* format. Select a Mask of regions
  to be excluded from the segmentation, such as the result from the Threshold module.
  Only regions that are not blocked by the mask will be segmented. For this reason,
  additional modules such as *ImageMath*, *RemoveHoles*, *Closing*, etc. may be helpful
  for refining the result from the *Threshold* module into a good mask.

When working with 3D image stacks, the objects of interest will sometimes be completely
out of focus and absent in the top and/or bottom planes of the stack. When setting up
your Threshold modules, you may need to set appropriate minimum threshold limits to
prevent the algorithm from trying to detect noise when no objects are present,
particularly if using local thresholding. ‘Empty’ z-slices should ideally end up being
masked off entirely.

Although the resulting objects will be visualised in CellProfiler windows as a series
of 2D slices, measurements are performed by treating them as solid, 3D objects.
Subsequent modules such as *MeasureObjectSizeShape* will therefore report some different
features, such as Volume instead of Area.

.. _3D-monolayer tutorial: https://github.com/CellProfiler/tutorials/tree/master/3d_monolayer
.. _MATLAB blog post: https://www.mathworks.com/company/newsletters/articles/the-watershed-transform-strategies-for-image-segmentation.html