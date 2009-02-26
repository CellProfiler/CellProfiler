function scales=CPgetpriorscales(handles, CurrentModuleNum)

%
% This function scans back over the modules prior to the current module
% to get a list of texture, neighbor radial distribution or image quality
% scales, returning them in an array.
%
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

if (nargin < 2)
    [CurrentModule, CurrentModuleNum] = CPwhichmodule(handles);
end
scales = [];
for i = 1:(CurrentModuleNum-1)
    ModuleName=handles.Settings.ModuleNames(i);
    if strcmp(ModuleName,'MeasureTexture')
        scales=add_scale(scales, str2double(handles.Settings.VariableValues(i,8)));
    elseif strcmp(ModuleName,'MeasureRadialDistribution')
        for bin_number=1:str2double(handles.Settings.VariableValues(i,4))
            scales=add_scale(scales, bin_number);
        end
    elseif strcmp(ModuleName,'MeasureObjectNeighbors')
        scales=add_scale(scales,str2double(handles.Settings.VariableValues(i,2)));
    elseif strcmp(ModuleName,'MeasureImageQuality')
        scales=add_scale(scales,str2double(handles.Settings.VariableValues(i,2)));
    end
end
scales = sort(scales);
end

function result=add_scale(scales,value)
    if isempty(find(scales == value, 1))
        result=[scales, value];
    else
        result=scales;
    end
end