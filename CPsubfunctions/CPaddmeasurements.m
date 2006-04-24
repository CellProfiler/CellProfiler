function handles = CPaddmeasurements(handles,Object,Measure,Feature,Data)

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
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2802 $

FeaturesField = [Measure,'Features'];

if isfield(handles.Measurements.(Object),FeaturesField)
    if handles.Current.SetBeingAnalyzed == 1
        CheckForRedundant = strmatch(Feature,handles.Measurements.(Object).(FeaturesField));
        if length(CheckForRedundant) > 1
            error('Image processing was canceled because you are attempting to create the same measurements, please remove redundant module.');
        end
        NewColumn = length(handles.Measurements.(Object).(FeaturesField)) + 1;
        handles.Measurements.(Object).(FeaturesField)(NewColumn) = {Feature};
        handles.Measurements.(Object).(Measure){handles.Current.SetBeingAnalyzed}(:,NewColumn) = Data;
    else
        OldColumn = strmatch(Feature,handles.Measurements.(Object).(FeaturesField));
        if length(OldColumn) > 1
            error('Image processing was canceled because you are attempting to create the same measurements, please remove redundant module.');
        elseif isempty(OldColumn)
            error('This should not happen. Please look at code for CPaddmeasurements. OldColumn is empty and it is not the first set being analyzed.');
        end
        handles.Measurements.(Object).(Measure){handles.Current.SetBeingAnalyzed}(:,OldColumn) = Data;
    end
else
    handles.Measurements.(Object).(FeaturesField) = {Feature};
    handles.Measurements.(Object).(Measure){handles.Current.SetBeingAnalyzed} = Data;
end