function CPcheckplatedivisibility(handles.Current.NumberOfImageSets,NumberOfCyclesPerPlate);

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision: 5025 $

%%% Checks whether the total number of image cycles to be processed is
%%% evenly divisible by the number of image cycles per plate. Issues a
%%% warning if not (a warning that does not cancel processing).

Divisibility = rem(handles.Current.NumberOfImageSets,NumberOfCyclesPerPlate);
if  Divisibility~=0
    CPwarndlg(['Given your specifications for numbers of rows, columns, and image cycles per well, the number of image cycles you have chosen works out to ', num2str(handles.Current.NumberOfImageSets/NumberOfCyclesPerPlate),' plates, which is not an integer. You may want to check that your settings and/or image files are correct.']);
end