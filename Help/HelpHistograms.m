function HelpHistograms

% HISTOGRAMS: 
% The individual cell measurements can be displayed in histogram
% format using this button.  As prompted, select the output file
% containing the measurements, then choose the measurement parameter
% to be displayed. You will then have the option of loading names for
% each image so that each histogram you make will be labeled with
% those names (if the measurement file does not already have names
% embedded). If you choose to import sample names here, you will need
% to select a text file that contains one name for every sample, even
% if you only plan to view a subset of the image data as histograms.
% Next, you will be prompted to select which sample numbers to
% displa.To display data from one image, enter that image's number as
% both the first and last sample.  To display several images' data  as
% separate histograms, enter the first and last sample numbers as
% appropriate (the full range is entered by default).  To display all
% the cells' data from several images in a single histogram, enter the
% first and last sample numbers as appropriate. 
% 
% You may then choose the number of bins to be used, the minimum and
% maximum values to be displayed in the histogram (on the X axis),
% whether you want all the cells' data to be displayed in a single
% (cumulative) histogram or in separate histograms, and whether you
% want the Y axis (number of cells) to be absolute (the same for all
% histograms) or relative (scaled to fit the maximum value for that
% sample). It may take some time to then process the data.  
% 
% X axis labels for histograms: Typically, the X axis labels will be
% too crowded.  This default state is shown because you might want to
% know the exact values that were used for the histogram bins.  The
% actual numbers can be viewed by clicking the 'This window' button
% under 'Change plots' and looking at the numbers listed under
% 'Labels'.  To change the X axis labels, you can click 'Fewer' in the
% main histogram window, or you can click a button under 'Change
% plots' and either change the font size on the 'Style' tab, or check
% the boxes marked 'Auto' for 'Ticks' and 'Labels' on the 'X axis'
% tab. Be sure to check both boxes, or the labels will not be
% accurate. To revert to the original labels, click 'Restore' in the
% main histogram window, but beware that this function does not work
% when more than one histogram window is open at once, because the
% most recently produced histogram's labels will be used for everything.
% 
% Change plots/change bars buttons: These buttons allow you to change
% properties of the plots or the bars within the plots for either
% every plot in the window ('This window'), the current plot only
% ('Current'), or every plot inevery open window ('All windows').
% This include colors, axis limits and other properties.
% 
% Other notes about histograms: (1) Data outside the range you
% specified to calculate histogram bins are added together and
% displayed in the first and last bars of the histogram. (2) Only the
% display can be changed in this window, including axis limits.  The
% histogram bins themselves cannot be changed here because the data
% must be recalculated. (3) If a change you make using the 'Change
% display' buttons does not seem to take effect in all of the desired
% windows, try pressing enter several times within that box, or look
% in the bottom of the Property Editor window that opens when you
% first press one of those buttons.  There may be a message describing
% why.  For example, you may need to deselect 'Auto' before changing
% the limits of the axes. (4) The labels for each bar specify the low
% bound for that bin.  In other words, each bar includes data equal to
% or greater than the label, but less than the label on the bar to its
% right.