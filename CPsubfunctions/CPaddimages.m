function handles = CPaddimages(handles, varargin)
% Add images to the handles.Pipeline structure.
% Location will be "handles.Pipeline.ImageName".

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
% $Revision: 5722 $

% Parse out varargin
if mod(length(varargin),2) ~= 0 || ...
   ~all(cellfun(@ischar,varargin(1:2:end)) & (cellfun(@isnumeric,varargin(2:2:end)) | cellfun(@islogical,varargin(2:2:end))))
    error('The argument list must be of the form: ''ImageName1'', ImageData1, etc');
else
    ImageName = varargin(1:2:end);
    ImageData = varargin(2:2:end);
end

CPvalidfieldname(ImageName);

% Checks have passed, add the data
for i = 1:length(ImageName)
    handles.Pipeline.(ImageName{i}) = ImageData{i};
end
