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
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

% Variable settings for PyCP
% I don't really understand why there is a 'to' and a 'from' for the email
% addresses, don't you always want to send the updates to yourself? Though
% I suppose you could also send them to another colleague as well.  Also..
% isn't the incoming and outgoing mail server different?? Maybe this
% confusion is all just my lack of understanding the magic of email, not
% the variables actually being confusing.

% Vars 04 & 05 could be:
% Send an email after:
% choice1: The first cycle is completed
% choice2: The last cycle is completed
% choice3: Send an email after a cycle I specify
% choice4: Send an email every Nth cycle
% as long as multiple selections are poss. (since someone could
% thoertically want some combination of these?)
% If they pick choice 3 and/or 4, then they would get text boxes asking for the cycle #
% Ideally instead of an arbitrary # of text boxes, the user could just
% click on a box that says 'Specify another cycle number'

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
CPclosefigure(handles,CurrentModule)