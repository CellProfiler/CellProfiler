function ImageToolWindow(handles)

% Help for the Image Tool Window:
% Category: Image Tools
%
% SHORT DESCRIPTION:
% The Image Tool Window opens when you click on any image and allows
% opening the image in a new window, displaying a pixel intensity
% histogram, measuring length in the image, changing the figure colormap,
% and saving the image.
% *************************************************************************
%
% The Image Tool Window contains these functions:
% 
% Open in new window - Opens the image in its own, fresh window.
%
% Histogram - Shows a pixel intensity histogram for the image.
%
% Measure Length - This tool creates a line in the image. By moving the
% ends of the line, you can measure distances in the image. Right-clicking
% the line reveals several options, including deleting the line. You can
% place multiple length-measuring lines on an image. Note that sometimes
% this line may interfere when saving the underlying image.
%
% Change Colormap - Opens a window that allows you to change the colormap
% of the selected figure. You can select the default colormap (which you
% can set under File > Set Preferences) or any other predetermined
% colormap. Note that the colormap selected will apply to all non-RGB
% images in the entire figure, and not only to the image selected. The
% Apply To All button will change the colormap in all module display
% windows and any other windows that contain images. If you are running the
% developer's version of CellProfiler, you can also open a colormap editor,
% which enables you to create personalized colormaps. It will modify the
% colormap of the last active figure, so be careful if you open it, click
% another figure and go back to it, because you might be changing the
% colormap of a figure you did not intend to change. See also Help >
% General Help > Colormaps.
%
% Save to Matlab workspace - If you are using Matlab Developer's version,
% this tool saves the image to the Matlab workspace with the variable name
% "Image". Be careful not to overwrite existing variables in your workspace
% using this tool.
%
% Save Image As - See the help for the Save Images module.
%
%
% Technical details:
% The CPimagetool function opens or updates the Image Tool window when the
% user clicks on an image produced by a module. The tool is embedded by the
% CPimagesc function which is used to display almost all images in
% CellProfiler.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%
% Website: http://www.cellprofiler.org

% This function itself is simply a menu item in the CellProfiler Image Tools
% menu which informs the user that more tools can be accessed by clicking
% on an individual image within a figure window. Its help is accessible via
% the normal image tools help extraction.
CPmsgbox('For more image tools, click on an individual image within a display window.')