function HelpShowDataOnImage

% Help for DISPLAY DATA ON IMAGE: 
% This allows you to extract measurements from an output file and
% overlay any measurements that you have made on any image. For
% example, you could look at the DNA content (e.g.
% IntegratedIntensityOrigBlue) of each cell on an image of nuclei.
% Or, you could look at cell area on an image of nuclei.  
% 
% First, you are asked to select the measurement you want to be
% displayed on the image.  Next, you are asked to select the X and
% then the Y locations where these measurements should be displayed.
% Typically, your options are the XY locations of the nuclei, or the
% XY locations of the cells, and these are usually named something
% like 'CenterXNuclei'.  If your output file has measurements from
% many images, you then select which sample number to view.
% 
% Then, CellProfilerTM tries to guide you to find the image that
% corresponds to this sample number.  First, it asks which file name
% would be most helpful for you to find the image. CellProfilerTM
% uses whatever you enter here to look up the exact file name you are
% looking for, so that you can browse to find the image. Once the
% image is selected, extraction ensues and eventually the image will
% be shown with the measurements on top.
% 
% You can use the tools at the top to zoom in on this image. If the
% text is overlapping and not easily visible, you can change the
% number of decimal places shown with the 'Fewer significant digits'
% button, or you can change the font size with the 'Text Properties'.
% You can also change the font style, color, and other properties with
% this button.  
% 
% The resulting figure can be saved in Matlab format (.fig) or
% exported in a traditional image file format.