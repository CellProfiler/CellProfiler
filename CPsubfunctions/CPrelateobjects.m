function [handles,ChildList,FinalParentList] = CPrelateobjects(handles,ChildName,ParentName,ChildLabelMatrix,ParentLabelMatrix)

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
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2802 $

%%% This line creates two rows containing all values for both label matrix
%%% images. It then takes the unique rows (no repeats), and sorts them
%%% according to the first column which is the sub object values.
ChildParentList = sortrows(unique([ChildLabelMatrix(:) ParentLabelMatrix(:)],'rows'),1);
%%% We want to get rid of the children values and keep the parent values.
ParentList = ChildParentList(:,2);
%%% This gets rid of all parent values which have no corresponding children
%%% values (where children = 0 but parent = 1).
for i = 1:max(ChildParentList(:,1))
    ParentValue = max(ParentList(ChildParentList(:,1) == i));
    if isempty(ParentValue)
        ParentValue = 0;
    end
    FinalParentList(i,1) = ParentValue;
end

if exist('FinalParentList','var')
    if max(ChildLabelMatrix(:)) ~= size(FinalParentList,1)
        error('Image processing was canceled in CPrelateobjects because secondary objects cannot have two parents, something is wrong.');
    end
    handles = CPaddmeasurements(handles,ChildName,'Parent',ParentName,FinalParentList);
end

for i = 1:max(ParentList)
    if exist('FinalParentList','var')
        ChildList(i,1) = length(FinalParentList(FinalParentList == i));
    else
        ChildList(i,1) = 0;
    end
end

if exist('ChildList','var')
    handles = CPaddmeasurements(handles,ParentName,'Children',[ChildName,' Count'],ChildList);
else
    ChildList = [];
end

if ~exist('FinalParentList','var')
    FinalParentList = [];
end