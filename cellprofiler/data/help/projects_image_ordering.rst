Image Ordering
==============

Files are processed on a Mac or PC by the `ASCII code order`_, in ascending numerical order of the `ASCII codes`_ that represent leading characters, including the following rules:

Leading-number files are processed first (eg 0019.jpg before Image0019.jpg or image0019.jpg). Leading-number files are processed in numerical order.
Leading-capital letter files are processed before leading-lowercase named files (Image0019.jpg before image0019.jpg). Leading-capital letter files are processed in alphabetical order first (A002.jpg before B001.jpg) and then by numerical order (A001.jpg before A002.jpg).
Leading-lowercase letter named files are processed in numerical order.
Trailing letters/numbered files are processed following the same rules, after all the leading letter/number rules have been applied. (Plate01B.jpg before Plate01a.jpg)
Note: By the above rules, CellProfiler will process files according to the first digit (so Image15.jpg comes before Image2.jpg). To fix this, make sure the digits are standardized (Image02.jpg is processed before Image15.jpg). This should be the case for images coming from a microscope, but if you name your folders manually, be careful!

Folders are processed according to the same rules, (eg ALL images within Plate01 are processed before those within Plate02) and within folders file processing follows the same rules.

If making your platemap match the way your microscope names your images will be very difficult/confusing, it may be easier to rename your images.
To be safe when manually naming files or folders:

Stick with all uppercase, or all lowercase, so you don’t get confused.
Use leading zeroes on numbers (eg 000001)

.. _ASCII code order: https://en.wikipedia.org/wiki/ASCII#Order
.. _ASCII codes: https://en.wikipedia.org/wiki/ASCII#Printable_characters
