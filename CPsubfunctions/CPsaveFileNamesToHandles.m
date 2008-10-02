% CPSAVEFILENAMESTOHANDLES Save filenames and pathnames.
%   This function will move the subdirectory part of the filename to
%   the pathname before saving to the measurement part of the handles
%   structure.
%
%   Example:
%      handles = CPsaveFileNamesToHandles(handles, ImageName, Pathname, ...
%					  FileNames)
%
%   ImageName should be a cell array containing 'OrigBlue', 'OrigGreen', etc.
%   BasePathName should be a string, e.g., '/tmp/foo'.
%   FileNames should be a cell array containing 'A01.TIF', 'b/A02.TIF', etc.
function handles = CPsaveFileNamesToHandles(handles, ImageName, ...
					    BasePathName, FileNames)
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2008.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$
for i = 1:length(ImageName)
    [subpart, filenamepart, ext] = fileparts(FileNames{i});
    pathname = fullfile(BasePathName, subpart);
    handles = CPaddmeasurements(handles, 'Image', ...
				['FileName_', ImageName{i}], ...
				[filenamepart ext]);  
    handles = CPaddmeasurements(handles, 'Image', ...
				['PathName_', ImageName{i}], ...
				pathname);
end
