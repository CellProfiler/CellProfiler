function HelpNormalization

% Help for NORMALIZATION:
% If there is some sort of artifact that causes a systematic shift in
% your measurements depending on where the sample was physically
% located in the layout (e.g. certain parts of the slide were more
% brightly illuminated), you can calculate normalization factors to
% correct your data using this button.  You will be prompted first to
% copy the measurements to the clipboard.  This should be a single
% column of data from a set of images. Then, use Matlab's import
% wizard to import your data and press 'Finish' before you click the
% OK button in the tiny prompt window. You will then be asked about
% the physical layout of the samples (rows and columns). The
% normalization assumes that most of the samples should in reality
% equal values, with only a few being real hits. Therefore, you can
% enter the percentile (range = 0 to 1) below and above which values
% will be excluded from the normalization correction.  This should
% basically exclude the percentage of samples that are likely to be
% real hits.
% 
% The results of the calculation are displayed in an output window.
% The imported data is shown on the left: the range of colors
% indicates the range of values, as indicated by the bar immediately
% to the right of the plot.  The central plot shows the ignored
% samples. Usually the ignored samples are shown as blue (these are
% the samples that were in the top or bottom percentile that you
% specified, and were not used to fit the smooth correction function).
% The plot on the right, with its corresponding color bar, shows the
% smoothened, calculated correction factors.  You can change the order
% of the polynomial which is used to fit to the data by opening the
% main CellProfiler program file (CellProfiler.m) and adjusting the
% calculation towards the end of the subfunction called
% 'NormalizationButton'. (Save a backup elsewhere first, and do not
% change the name of the main CellProfiler program file or problems
% will ensue.)
% 
% The correction factors (or, normalization factors) are placed onto
% the clipboard for you to paste next to the original data (e.g. in
% Excel). You can then divide the input values by these correction
% factors to yield the normalized data. After clicking OK, your
% original data is then placed on the clipboard so that, if desired,
% you can paste it next to the original data (e.g. in Excel) just to
% doublecheck that the proper numbers were used. Both sets of data are
% also displayed in the Matlab command window to copy and paste.