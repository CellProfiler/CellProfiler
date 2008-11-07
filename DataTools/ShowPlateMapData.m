function orig_handles = ShowPlateMapData(handles)

% Help for the Show Plate Map Data tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Shows measurement data for plate maps in heat-map form
% *************************************************************************
% This tool lets you display the data in a measurements file in plate map
% format. One image per image set should use FileNameMetadata or some other
% mechanism to capture the plate's name and the well row and column.
% The user selects:
% * the names of the measurements that provide the metadata information
% * the measurement to display
% * the minimum and maximum values for the measurement
% * whether to display the mean, median, maximum or minimum value
%   for the measurement.
%
% See also FileNameMetadata module

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
% $Revision$


%%% This tool shouldn't change anything in the handles.
orig_handles = handles;

%%% Asks the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = CPuigetfile('*.mat', 'Select the raw measurements file',handles.Current.DefaultOutputDirectory);
if RawFileName == 0,return,end

load(fullfile(RawPathname,RawFileName));

% Try to convert features
handles = CP_convert_old_measurements(handles);

Metadata = CPShowPlateMapDataGUI(handles);
CPShowPlateMapDataFig(Metadata,handles);
