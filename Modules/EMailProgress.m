function handles = AlgEMailProgress(handles)

% Help for the E-Mail Progress module:
% Category: ?
%
% This module e-mails the user-specified recipients about the current
% progress of the image processing algorithms as well as the expected time
% remaining until completion.  The user can specify how often e-mails are
% sent out (for example, after the first image set, after the last image
% set, after every X number of images, after Y images).  Note:  This
% module should be the last module loaded.  If it isn't the last module
% loaded, then the notifications are sent out after all the algorithms that
% are processed before this module is completed, but not before all the
% algorithms that comes after this module.

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
%
% $Revision 1.00 $

% PROGRAMMING NOTE
% HELP:
% The first unbroken block of lines will be extracted as help by
% CellProfiler's 'Help for this analysis module' button as well as
% Matlab's built in 'help' and 'doc' functions at the command line. It
% will also be used to automatically generate a manual page for the
% module. An example image demonstrating the function of the module
% can also be saved in tif format, using the same name as the
% algorithm (minus Alg), and it will automatically be included in the
% manual page as well.  Follow the convention of: purpose of the
% module, description of the variables and acceptable range for each,
% how it works (technical description), info on which images can be 
% saved, and See also CAPITALLETTEROTHERALGORITHMS. The license/author
% information should be separated from the help lines with a blank
% line so that it does not show up in the help displays.  Do not
% change the programming notes in any modules! These are standard
% across all modules for maintenance purposes, so anything
% module-specific should be kept separate.

% PROGRAMMING NOTE
% DRAWNOW:
% The 'drawnow' function allows figure windows to be updated and
% buttons to be pushed (like the pause, cancel, help, and view
% buttons).  The 'drawnow' function is sprinkled throughout the code
% so there are plenty of breaks where the figure windows/buttons can
% be interacted with.  This does theoretically slow the computation
% somewhat, so it might be reasonable to remove most of these lines
% when running jobs on a cluster where speed is important.
drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

% PROGRAMMING NOTE
% VARIABLE BOXES AND TEXT: 
% The '%textVAR' lines contain the text which is displayed in the GUI
% next to each variable box. The '%defaultVAR' lines contain the
% default values which are displayed in the variable boxes when the
% user loads the algorithm. The line of code after the textVAR and
% defaultVAR extracts the value that the user has entered from the
% handles structure and saves it as a variable in the workspace of
% this algorithm with a descriptive name. The syntax is important for
% the %textVAR and %defaultVAR lines: be sure there is a space before
% and after the equals sign and also that the capitalization is as
% shown.  Don't allow the text to wrap around to another line; the
% second line will not be displayed.  If you need more space to
% describe a variable, you can refer the user to the help file, or you
% can put text in the %textVAR line above or below the one of
% interest, and do not include a %defaultVAR line so that the variable
% edit box for that variable will not be displayed; the text will
% still be displayed. CellProfiler is currently being restructured to
% handle more than 11 variable boxes. Keep in mind that you can have
% several inputs into the same box: for example, a box could be
% designed to receive two numbers separated by a comma, as long as you
% write a little extraction algorithm that separates the input into
% two distinct variables.  Any extraction algorithms like this should
% be within the VARIABLES section of the code, at the end.

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2double(handles.currentalgorithm);

%textVAR01 = Send an e-mail after the first image set is completed?
%defaultVAR01 = Y
FirstImageEmail = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = Send an e-mail after the last image set is completed?
%defaultVAR02 = Y
LastImageEmail = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = Send an e-mail after every X number of image sets?
%defaultVAR03 = 0
EveryXImageEmail = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,3}));

%textVAR04 = Send an e-mail after _ number of image sets?
%defaultVAR04 = 0
Specified1Email = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,4}));

%textVAR05 = Send an e-mail after _ number of image sets?
%defaultVAR05 = 0
Specified2Email = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,5}));

%textVAR06 = Send an e-mail after _ number of image sets?
%defaultVAR06 = 0
Specified3Email = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,6}));

%textVAR07 = Send an e-mail after _ number of image sets?
%defaultVAR07 = 0
Specified4Email = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,7}));

%textVAR08 = Send an e-mail after _ number of image sets?
%defaultVAR08 = 0
Specified5Email = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,8}));

%textVAR09 = Please enter the SMTP server
%defaultVAR09 = mail
SMTPServer = char(handles.Settings.Vvariable{CurrentAlgorithmNum,9});

%textVAR10 = Send e-mails from this e-mail address
%defaultVAR10 = youremail@address.com
AddressFrom = char(handles.Settings.Vvariable{CurrentAlgorithmNum,10});

%textVAR11 = Send e-mails to these e-mail addresses
%defaultVAR11 = default@default.com
AddressEmail = char(handles.Settings.Vvariable{CurrentAlgorithmNum,11});



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SENDING EMAILS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
setpref('Internet','SMTP_Server',SMTPServer);
setpref('Internet','E_mail',AddressFrom);

SetBeingAnalyzed = handles.setbeinganalyzed;

numAddresses = 0
while (isempty(AddressEmail) == 0)
    numAddresses = numAddresses + 1
    [AddressTo{numAddresses} AddressEmail] = strtok(AddressEmail, ',');
end


if ( strcmp(FirstImageEmail, 'Y') & (SetBeingAnalyzed == 1))
    subject = 'CellProfiler:  First Image Set has been completed';
    sendmail(AddressTo,subject,subject);
elseif ( strcmp(LastImageEmail, 'Y') & SetBeingAnalyzed == handles.Vnumberimagesets)
    subject = 'CellProfiler:  Last Image Set has been completed';
    sendmail(AddressTo,'CellProfiler Progress',subject);
elseif ( (EveryXImageEmail > 0) & ( mod(SetBeingAnalyzed,EveryXImageEmail) == 0) )
    subject = 'CellProfiler: Last Image Set has been completed';
    sendmail(AddressTo,subject,subject);
end
    

%%%%%%%%%%%%%%%%%%%%
%%% FIGURE WINDOW %%%
%%%%%%%%%%%%%%%%%%%%
drawnow

if SetBeingAnalyzed == 1
    %%% The figure window display is unnecessary for this module, so the figure
    %%% window is closed the first time through the module.
    %%% Determines the figure number.
    fieldname = ['figurealgorithm',CurrentAlgorithm];
    ThisAlgFigureNumber = handles.(fieldname);
    %%% If the window is open, it is closed.
    if any(findobj == ThisAlgFigureNumber) == 1;
        close(ThisAlgFigureNumber)
    end
end

drawnow