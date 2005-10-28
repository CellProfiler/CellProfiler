function handles = SendEmail(handles)

% Help for the Send E-Mail module:
% Category: Other
%
% This module e-mails the user-specified recipients about the current
% progress of the image processing modules as well as the expected time
% remaining until completion.  The user can specify how often e-mails are
% sent out (for example, after the first image set, after the last image
% set, after every X number of images, after Y images).  Note:  This
% module should be the last module loaded.  If it isn't the last module
% loaded, then the notifications are sent out after all the modules that
% are processed before this module is completed, but not before all the
% modules that comes after this module.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
%
% $Revision$

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = Send e-mails to these e-mail addresses
%defaultVAR01 = default@default.com
AddressEmail = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Send e-mails from this e-mail address
%defaultVAR02 = youremail@address.com
AddressFrom = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Enter the SMTP server
%defaultVAR03 = mail
SMTPServer = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Send an e-mail after the first image set is completed?
%choiceVAR04 = Yes
%choiceVAR04 = No
FirstImageEmail = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Send an e-mail after the last image set is completed?
%choiceVAR05 = Yes
%choiceVAR05 = No
LastImageEmail = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Send an e-mail after every X number of image sets?
%defaultVAR06 = 0
EveryXImageEmail = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Send an e-mail after _ number of image sets?
%defaultVAR07 = 0
Specified1Email = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = Send an e-mail after _ number of image sets?
%defaultVAR08 = 0
Specified2Email = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,8}));

%textVAR09 = Send an e-mail after _ number of image sets?
%defaultVAR09 = 0
Specified3Email = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,9}));

%textVAR10 = Send an e-mail after _ number of image sets?
%defaultVAR10 = 0
Specified4Email = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,10}));

%textVAR11 = Send an e-mail after _ number of image sets?
%defaultVAR11 = 0
Specified5Email = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,11}));

%%%VariableRevisionNumber = 01

%%%%%%%%%%%%%%%%%%%%%
%%% SENDING EMAILS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

setpref('Internet','SMTP_Server',SMTPServer);
setpref('Internet','E_mail',AddressFrom);

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

numAddresses = 0;
while isempty(AddressEmail) == 0
    numAddresses = numAddresses + 1;
    [AddressTo{numAddresses} AddressEmail] = strtok(AddressEmail, ',');
end

if strcmp(FirstImageEmail, 'Yes') == 1 & SetBeingAnalyzed == handles.Current.StartingImageSet
    subject = 'CellProfiler:  First image set has been completed';
    if (SetBeingAnalyzed > 1)
        subject = [subject,' after Restart, image number ',num2str(SetBeingAnalyzed)];
    end
    sendmail(AddressTo,subject,subject);
elseif strcmp(LastImageEmail, 'Yes') == 1 & SetBeingAnalyzed == handles.Current.NumberOfImageSets
    subject = 'CellProfiler:  Last image set has been completed';
    sendmail(AddressTo,'CellProfiler Progress',subject);
elseif EveryXImageEmail > 0 & mod(SetBeingAnalyzed,EveryXImageEmail) == 0
    subject = ['CellProfiler: ', num2str(SetBeingAnalyzed),' image sets have been completed'];
    sendmail(AddressTo,subject,subject);
elseif Specified1Email == SetBeingAnalyzed | Specified2Email == SetBeingAnalyzed | Specified3Email == SetBeingAnalyzed | Specified4Email == SetBeingAnalyzed | Specified5Email == SetBeingAnalyzed
    subject = ['CellProfiler: ', num2str(SetBeingAnalyzed),' image sets have been completed'];
    sendmail(AddressTo,subject,subject);
end

%%%%%%%%%%%%%%%%%%%%
%%% FIGURE WINDOW %%%
%%%%%%%%%%%%%%%%%%%%
drawnow

if SetBeingAnalyzed == handles.Current.StartingImageSet
    %%% The figure window display is unnecessary for this module, so the figure
    %%% window is closed the first time through the module.
    %%% Determines the figure number.
    fieldname = ['FigureNumberForModule',CurrentModule];
    ThisModuleFigureNumber = handles.Current.(fieldname);
    %%% If the window is open, it is closed.
    if any(findobj == ThisModuleFigureNumber) == 1;
        close(ThisModuleFigureNumber)
    end
end

drawnow

%%% STANDARD HELP STATEMENTS THAT ARE UNNECESSARY FOR THIS MODULE:
