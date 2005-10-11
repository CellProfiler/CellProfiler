function varargout = errordlg(ErrorString,DlgName,Replace)
%ERRORDLG Error dialog box.
%  HANDLE = ERRORDLG(ErrorString,DlgName,CREATEMODE) creates an 
%  error dialog box which displays ErrorString in a window 
%  named DlgName.  A pushbutton labeled OK must be pressed 
%  to make the error box disappear.  
%
%  ErrorString will accept any valid string input but a cell 
%  array is preferred.
%
%  ERRORDLG uses MSGBOX.  Please see the help for MSGBOX for a
%  full description of the input arguments to ERRORDLG.
%  
%  See also MSGBOX, HELPDLG, QUESTDLG, WARNDLG.

%  Author: L. Dean
%  Copyright 1984-2002 The MathWorks, Inc.
%  $Revision: 5.24 $  $Date: 2002/04/15 03:25:03 $

NumArgIn = nargin;
if NumArgIn==0,
   ErrorString = {'This is the default error string.'};
end

if NumArgIn<2,  DlgName = 'Error Dialog'; end
if NumArgIn<3,  Replace='non-modal'     ; end

% Backwards Compatibility
if ischar(Replace),
  if strcmp(Replace,'on'),
    Replace='replace';
  elseif strcmp(Replace,'off'),
    Replace='non-modal';
  end
end

if ischar(ErrorString) & ~iscellstr(ErrorString)
    ErrorString = cellstr(ErrorString);
end
if ~iscellstr(ErrorString)
    error('Errorstring should be a string or cell array of strings');
end

ErrorStringCell = cell(0);
for i = 1:length(ErrorString)
    ErrorStringCell{end+1} = xlate(ErrorString{i});
end

handle = msgbox(ErrorStringCell,DlgName,'error',Replace);
set(handle,'color',[.7 .7 .9]);
if nargout==1,varargout(1)={handle};end