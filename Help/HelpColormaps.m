function HelpColormaps
helpdlg(help('HelpColormaps'))

% Default colormaps can be set in File > Set preferences.
%
% Label colormap - affects how objects are colored. Colorcube (and possibly
% other colormaps) is not recommended because some intensity values are
% displayed as black. Jet is the default.
%
% Intensity colormap - affects how grayscale images are displayed.
% Colorcube (and possibly other colormaps) is not recommended because some
% intensity values are displayed as black. Gray is recommended.
%
% Choose from these colormaps:
% autumn bone colorcube cool copper flag gray hot hsv jet lines pink
% prism spring summer white winter

%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.