function handles = SendEmail(handles)

% Help for the Send Email module:
% Category: Other
%
% SHORT DESCRIPTION:
% Sends emails to a specified address at desired stages of the processing.
% *************************************************************************
%
% This module emails the user-specified recipients about the current
% progress of the image processing. The user can specify how often emails
% are sent out (for example, after the first cycle, after the last cycle,
% after every N cycles, after N cycles). This module should be placed at
% the point in the pipeline when you want the emails to be sent. If email
% sending fails for any reason, a warning message will appear but
% processing will continue regardless.
%
% Settings:
% Address to: you can send messages to multiple email addresses by entering
% them with commas in between.
%
% SMTP server: often the default 'mail' will work. If not, ask your network
% administrator for your outgoing mail server, which is often made up of
% part of your email address, e.g., Something@company.com. You might be
% able to find this information by checking your settings/preferences in
% whatever email program you use.

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
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles); %#ok Ignore MLint

%textVAR01 = Send e-mails to these e-mail addresses
%defaultVAR01 = default@default.com
AddressEmail = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Send e-mails from this e-mail address
%defaultVAR02 = youremail@address.com
AddressFrom = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Enter the SMTP server
%defaultVAR03 = mail
SMTPServer = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Send an e-mail after the first cycle is completed?
%choiceVAR04 = Yes
%choiceVAR04 = No
FirstImageEmail = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Send an e-mail after the last cycle is completed?
%choiceVAR05 = Yes
%choiceVAR05 = No
LastImageEmail = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Send an e-mail after every Nth cycle:
%defaultVAR06 = 0
EveryXImageEmail = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Send an e-mail after cycle number:
%defaultVAR07 = 0
Specified1Email = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = Send an e-mail after cycle number:
%defaultVAR08 = 0
Specified2Email = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,8}));

%textVAR09 = Send an e-mail after cycle number:
%defaultVAR09 = 0
Specified3Email = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,9}));

%textVAR10 = Send an e-mail after cycle number:
%defaultVAR10 = 0
Specified4Email = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,10}));

%textVAR11 = Send an e-mail after cycle number:
%defaultVAR11 = 0
Specified5Email = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,11}));

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%
%%% SENDING EMAILS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

setpref('Internet','SMTP_Server',SMTPServer);
setpref('Internet','E_mail',AddressFrom);

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

numAddresses = 0;
while isempty(AddressEmail) == 0
    numAddresses = numAddresses + 1;
    [AddressTo{numAddresses} AddressEmail] = strtok(AddressEmail, ','); %#ok Ignore MLint
end

try
    if strcmp(FirstImageEmail, 'Yes') && SetBeingAnalyzed == handles.Current.StartingImageSet
        subject = 'CellProfiler:  First cycle has been completed';
        if (SetBeingAnalyzed > 1)
            subject = [subject,' after Restart, image # ',num2str(SetBeingAnalyzed)];
        end
        sendmail(AddressTo,subject,subject);
    elseif strcmp(LastImageEmail, 'Yes') && SetBeingAnalyzed == handles.Current.NumberOfImageSets
        subject = 'CellProfiler:  Last cycle has been completed';
        sendmail(AddressTo,'CellProfiler Progress',subject);
    elseif EveryXImageEmail > 0 && mod(SetBeingAnalyzed,EveryXImageEmail) == 0
        subject = ['CellProfiler: ', num2str(SetBeingAnalyzed),' cycles have been completed'];
        sendmail(AddressTo,subject,subject);
    elseif Specified1Email == SetBeingAnalyzed || Specified2Email == SetBeingAnalyzed || Specified3Email == SetBeingAnalyzed || Specified4Email == SetBeingAnalyzed || Specified5Email == SetBeingAnalyzed
        subject = ['CellProfiler: ', num2str(SetBeingAnalyzed),' cycles have been completed'];
        sendmail(AddressTo,subject,subject);
    end
catch
    CPwarndlg('Your current settings in the SendEmail module did not allow email to be sent, but processing will continue.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% FIGURE WINDOW %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    if any(findobj == ThisModuleFigureNumber)
        close(ThisModuleFigureNumber)
    end
end