:orphan:

Why Use CellProfiler?
=====================

Most laboratories studying biological processes and human disease use
light/fluorescence microscopes to image cells and other biological
samples. There is strong and growing demand for software to analyze
these images, as automated microscopes collect images faster than can be
examined by eye and the information sought from images is increasingly
quantitative and complex.

CellProfiler is a versatile, open-source software tool for quantifying
data from biological images, particularly in high-throughput
experiments. CellProfiler is designed for modular, flexible,
high-throughput analysis of images, measuring size, shape, intensity,
and texture of every cell (or other object) in every image. Using the
point-and-click graphical user interface (GUI), users construct an image
analysis “pipeline”, a sequential series of modules that each perform an
image processing function such as illumination correction, object
identification (segmentation), and object measurement. Users mix and
match modules and adjust their settings to measure the phenotype of
interest. While originally designed for high-throughput images, it is
equally appropriate for low-throughput assays as well (i.e., assays of <
100 images).

CellProfiler can extract valuable biological information from images
quickly while increasing the objectivity and statistical power of
assays. It helps researchers approach a variety of biological questions
quantitatively, including standard assays (e.g., cell count, size,
per-cell protein levels) as well as complex morphological assays (e.g.,
cell/organelle shape or subcellular patterns of DNA or protein
staining).

The wide variety of measurements produced by CellProfiler serves as
useful “raw material” for machine learning algorithms. CellProfiler’s
companion software, CellProfiler Analyst, has an interactive machine
learning tool called Classifier which can learn to recognize a phenotype
of interest based on your guidance. Once you complete the training
phase, CellProfiler Analyst will score every object in your images based
on CellProfiler’s measurements. CellProfiler Analyst also contains tools
for the interactive visualization of the data produced by CellProfiler.

In summary, CellProfiler contains:

-  Advanced algorithms for image analysis that are able to accurately
   identify crowded cells and non-mammalian cell types.
-  A modular, flexible design allowing analysis of new assays and
   phenotypes.
-  Open-source code so the underlying methodology is known and can be
   modified or improved by others.
-  A user-friendly interface.
-  The ability to use high-throughput computing (clusters, cloud).
-  A design that eliminates the tedium of the many steps typically
   involved in image analysis, many of which are not easily transferable
   from one project to another (for example, image formatting, combining
   several image analysis steps, or repeating the analysis with slightly
   different parameters).

References
^^^^^^^^^^

For a full list of references, visit our `citation`_ page.

-  McQuin C, Goodman A, Chernyshev V, Kamentsky L, Cimini BA, Karhohs KW,
   Doan M, Ding L, Rafelski SM, Thirstrup D, Wiegraebe W. (2018)
   "CellProfiler 3.0: Next-generation image processing for biology."
   *PLoS biology* 16(7), e2005970
   (`link <https://doi.org/10.1371/journal.pbio.2005970>`__)
-  Kamentsky L, Jones TR, Fraser A, Bray MA, Logan D, Madden K, Ljosa V,
   Rueden C, Harris GB, Eliceiri K, Carpenter AE (2011) “Improved
   structure, function, and compatibility for CellProfiler: modular
   high-throughput image analysis software” *Bioinformatics*
   27(8):1179-1180
   (`link <https://doi.org/10.1093/bioinformatics/btr095>`__)
-  Carpenter AE, Jones TR, Lamprecht MR, Clarke C, Kang IH, Friman O,
   Guertin DA, Chang JH, Lindquist RA, Moffat J, Golland P, Sabatini DM
   (2006) “CellProfiler: image analysis software for identifying and
   quantifying cell phenotypes” *Genome Biology* 7:R100 (`link`_)
-  Lamprecht MR, Sabatini DM, Carpenter AE (2007) “CellProfiler: free,
   versatile software for automated biological image analysis”
   *Biotechniques* 42(1):71-75.
   (`link <https://doi.org/10.2144/000112257>`__)
-  Jones TR, Carpenter AE, Lamprecht MR, Moffat J, Silver S, Grenier J,
   Root D, Golland P, Sabatini DM (2009) “Scoring diverse cellular
   morphologies in image-based screens with iterative feedback and
   machine learning” *PNAS* 106(6):1826-1831
   (`link <https://doi.org/10.1073/pnas.0808843106>`__)
-  Jones TR, Kang IH, Wheeler DB, Lindquist RA, Papallo A, Sabatini DM,
   Golland P, Carpenter AE (2008) “CellProfiler Analyst: data
   exploration and analysis software for complex image-based screens”
   *BMC Bioinformatics* 9(1):482
   (`link <https://doi.org/10.1186/1471-2105-9-482>`__)

.. _citation: http://cellprofiler.org/citations/
.. _link: https://doi.org/10.1186/gb-2006-7-10-r100
