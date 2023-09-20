Plate Viewer
============

The plate viewer displays the images in your
experiment in plate format. Your project must define an image set list
with metadata annotations for the image’s well and, optionally its plate
and site. The plate viewer will then group your images by well and
display a plate map for you. If you have defined a plate metadata tag
(with the name, “Plate”), the plate viewer will group your images by
plate and display a choice box that lets you pick the plate to display.

Click on a well to see the images for that well. If you have more than
one site per well and have site metadata (with the name, “Site”), the
plate viewer will tile the sites when displaying, and the values under
“X” and “Y” determine the position of each site in the tiled grid.

The values for “Red”, “Green”, and “Blue” in each row are brightness
multipliers- changing the values will determine the color and scaling
used to display each channel. “Alpha” determines the weight each channel
contributes to the summed image.
