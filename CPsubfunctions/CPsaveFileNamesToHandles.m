% CPSAVEFILENAMESTOHANDLES Save filenames and pathnames.
%   Example:
%      handles = CPsaveFileNamesToHandles(handles, ImageName, Pathname, ...
%					  FileNames)
%
%   ImageName should be a cell array containing 'OrigBlue', 'OrigGreen', etc.
%   Pathname should be a string, e.g., '/tmp/foo'.
%   FileNames should be a cell array containing 'A01.TIF', 'A02.TIF', etc.
function handles = CPsaveFileNamesToHandles(handles, ImageName, Pathname, ...
					  FileNames)
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
% $Revision: 5228 $
for i = 1:length(ImageName)
    handles = CPaddmeasurements(handles, 'Image', ...
				['FileName_', ImageName{i}], ...
				FileNames{i});  
    handles = CPaddmeasurements(handles, 'Image', ...
				['PathName_', ImageName{i}], ...
				Pathname);
end
