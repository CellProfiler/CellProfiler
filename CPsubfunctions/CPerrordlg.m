function varargout = CPerrordlg(ErrorString,DlgName,Replace)
%CPERRORDLG Error dialog box.
%  HANDLE = CPERRORDLG(ErrorString,DlgName,CREATEMODE) creates an 
%  error dialog box which displays ErrorString in a window 
%  named DlgName.  A pushbutton labeled OK must be pressed 
%  to make the error box disappear.  
%
%  CPERRORDLG uses CPMSGBOX.  Please see the help for CPMSGBOX for a
%  full description of the input arguments to CPERRORDLG.

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

NumArgIn = nargin;
if NumArgIn==0,
   ErrorString = {'This is the default error string.'};
end

if NumArgIn<2,  DlgName='Error Dialog'; end
if NumArgIn<3,  Replace='non-modal'     ; end

errorinfo = lasterror;
if isfield(errorinfo, 'stack'),
    try
        stack = errorinfo.stack;
    catch
        %%% The line stackinfo = errorinfo.stack(1,1); will fail if the
        %%% errorinfo.stack is empty, which sometimes happens during
        %%% debugging, I think. So we catch it here.
        stack = {};
    end
end

ErrorCells = {ErrorString};

if size(stack, 1) > 0,
    ErrorCells{end+1} = '';
    ErrorCells{end+1} = 'Stack:';
    for index = 1:size(stack, 1),
        stackinfo = stack(index, 1);
        ErrorCells{end+1} = [stackinfo.name, ' in ', stackinfo.file, ' (', num2str(stackinfo.line) ')'];
    end
end

handle = CPmsgbox(ErrorCells,DlgName,'error',Replace);
if nargout==1,varargout(1)={handle};end
