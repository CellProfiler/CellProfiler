function varargout = CPhelpdlg(HelpString,DlgName)
%HELPDLG Help dialog box.
%  HANDLE = HELPDLG(HELPSTRING,DLGNAME) displays the 
%  message HelpString in a dialog box with title DLGNAME.  
%  If a Help dialog with that name is already on the screen, 
%  it is brought to the front.  Otherwise a new one is created.
%
%  HelpString will accept any valid string input but a cell
%  array is preferred.
%
%  See also MSGBOX, QUESTDLG, ERRORDLG, WARNDLG.

%  Author: L. Dean
%  Copyright 1984-2002 The MathWorks, Inc.
%  $Revision$  $Date$

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
% $Revision$

if nargin==0,
   HelpString ={'This is the default help string.'};
end
if nargin<2,
   DlgName = 'Help Dialog';
end

if ischar(HelpString) & ~iscellstr(HelpString)
    HelpString = cellstr(HelpString);
end
if ~iscellstr(HelpString)
    error('HelpString should be a string or cell array of strings');
end

HelpStringCell = cell(0);
for i = 1:length(HelpString)
    HelpStringCell{end+1} = xlate(HelpString{i});
end

handle = CPmsgbox(HelpStringCell,DlgName,'help','replace');

if nargout==1,varargout(1)={handle};end