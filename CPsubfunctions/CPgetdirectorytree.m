function Directories = CPgetdirectorytree(RootDirectory)
% Return a cell array of all directories under the root.
% RootDirectory is a single string.

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

Listing = sort(CPgetdirectories(RootDirectory));
% Delete any hidden directories (and "." and "..")
Listing(strncmp(Listing,'.',1)) = [];
Directories = [{RootDirectory}];
for i=1:length(Listing)
    SubDirectory = fullfile(RootDirectory,Listing{i});
    SubListing = CPgetdirectorytree(SubDirectory);
    Directories = [Directories;SubListing];
end