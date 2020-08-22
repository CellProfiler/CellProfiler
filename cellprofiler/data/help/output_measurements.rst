How Measurements are Named
==========================

In CellProfiler, measurements are exported as well as stored internally
using the following general nomenclature:
``MeasurementType_Category_SpecificFeatureName_Parameters``

Below is the description for each of the terms:

-  ``MeasurementType``: The type of data contained in the measurement,
   which can be one of three forms:

   -  *Per-image:* These measurements are image-based (e.g., thresholds,
      counts) and are specified with the name “Image” or with the
      measurement (e.g., “Mean”) for per-object measurements aggregated
      over an image.
   -  *Per-object:* These measurements are per-object and are specified
      as the name given by the user to the identified objects (e.g.,
      “Nuclei” or “Cells”).
   -  *Experiment:* These measurements are produced for a particular
      measurement across the entire analysis run (e.g., Z’ factors), and
      are specified with the name “Experiment”. See
      **CalculateStatistics** for an example.

-  ``Category:`` Typically, this information is specified in one of two
   ways:

   -  A descriptive name indicative of the type of measurement taken
      (e.g., “Intensity”)
   -  No name if there is no appropriate ``Category`` (e.g., if the
      *SpecificFeatureName* is “Count”, no ``Category`` is specified).

-  ``SpecificFeatureName:`` The specific feature recorded by a module
   (e.g., “Perimeter”). Usually the module recording the measurement
   assigns this name, but a few modules allow the user to type in the
   name of the feature (e.g., the **CalculateMath** module allows the
   user to name the arithmetic measurement).
-  ``Parameters:`` This specifier is to distinguish measurements
   obtained from the same objects but in different ways. For example,
   **MeasureObjectIntensity** can measure intensities for “Nuclei” in
   two different images. This specifier is used primarily for data
   obtained from an individual image channel specified by the **Images**
   module or a legacy **Load** module (e.g., “OrigBlue” and “OrigGreen”)
   or a particular spatial scale (e.g., under the category “Texture” or
   “Neighbors”). Multiple parameters are separated by underscores.

   Below are additional details specific to various modules:

   -  Measurements from the *AreaShape* and *Math* categories do not
      have a ``Parameter`` specifier.
   -  Measurements from *Intensity*, *Granularity*, *Children*,
      *RadialDistribution*, *Parent* and *AreaOccupied* categories will
      have an associated image as the Parameter.
   -  Measurements from the *Neighbors* and *Texture* category will
      have a spatial scale ``Parameter``.
   -  Measurements from the *Texture* and *RadialDistribution*
      categories will have both a spatial scale and an image
      ``Parameter``.
   -  Measurements from the *Texture* category will have a spacial 
      scale, image, and grayscale count ``Parameter``.

As an example, consider a measurement specified as
``Nuclei_Texture_DifferenceVariance_ER_3_256``:

-  ``MeasurementType`` is “Nuclei,” the name given to the detected
   objects by the user.
-  ``Category`` is “Texture,” indicating that the module
   **MeasureTexture** produced the measurements.
-  ``SpecificFeatureName`` is “DifferenceVariance,” which is one of the
   many texture measurements made by the **MeasureTexture** module.
-  There are three ``Parameters``, the first of which is “ER”. “ER” is the
   user-provided name of the image in which this texture measurement was
   made.
-  The second ``Parameter`` is “3”, which is the spatial scale at which
   this texture measurement was made, according to the user-provided
   settings for the module.
-  The final ``Parameter`` is "256", which is the number of gray levels
   used in calculating the texture.

See also the *Available measurements* heading under the main help for
many of the modules, as well as **ExportToSpreadsheet** and
**ExportToDatabase** modules.
