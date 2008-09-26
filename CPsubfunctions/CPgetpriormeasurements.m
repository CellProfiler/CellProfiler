function measurements=CPgetpriormeasurements(handles, CurrentModuleNum, ObjectOrImageName, CategoryName)

% Get the measurements prior to the given module number that are in the given
% category and for the given object or image.
%
% Note: The initial release of this returns all possible categories.
%
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
measurements={};
for i = 1:(CurrentModuleNum-1)
    SupportsMeasurements=false;
    for feature=handles.Settings.ModuleSupportedFeatures{i}
        if strcmp(feature,'measurements')
            SupportsMeasurements=true;
        end
    end
    if SupportsMeasurements
        handles.Current.CurrentModuleNumber=num2str(i);
        more_measurements=feval(handles.Settings.ModuleNames{i},...
            handles,'measurements',ObjectOrImageName, CategoryName);
        more_measurements=more_measurements(~ismember(more_measurements,measurements));
        measurements = [measurements, more_measurements];
    end
end

end