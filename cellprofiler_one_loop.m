% Help for the cellprofiler_one_loop function
%
% This function allows cluster analysis.

% The contents of this file are subject to the Mozilla Public License Version 
% 1.1 (the "License"); you may not use this file except in compliance with 
% the License. You may obtain a copy of the License at 
% http://www.mozilla.org/MPL/
% 
% Software distributed under the License is distributed on an "AS IS" basis,
% WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
% for the specific language governing rights and limitations under the
% License.
% 
% 
% The Original Code is the CellProfiler Cluster Main Loop.
% 
% The Initial Developer of the Original Code is
% Whitehead Institute for Biomedical Research
% Portions created by the Initial Developer are Copyright (C) 2003,2004
% the Initial Developer. All Rights Reserved.
% 
% Contributor(s):
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

% clear the results structure

clear handles_results
handles_results.CellProfilerError = [];

try
  cd(handles.RemoteImagePathName)
catch
  handles_results.CellProfilerError = ['Could not change directory to ' handles.RemoteImagePathName '.  Error was ' lasterr];
  return;
end
  

% this should be remotely set by the GUI
handles.setbeinganalyzed = setbeinganalyzed;

% clear the error variables
handles_results.CellProfilerError = [];

for SlotNumber = 1:handles.numAlgorithms,
  %%% If an algorithm is not chosen in this slot, continue on to the next.
  AlgNumberAsString = sprintf('%02d', SlotNumber);
  AlgName = ['Valgorithmname' AlgNumberAsString];
  if isfield(handles,AlgName) == 0
  else 
    %%% Saves the current algorithm number in the handles structure.
    handles.currentalgorithm = AlgNumberAsString;
    %%% The try/catch/end set catches any errors that occur during the
    %%% running of algorithm 1, notifies the user, breaks out of the image
    %%% analysis loop, and completes the refreshing process.
    try
      %%% Runs the appropriate algorithm, with the handles structure as an
      %%% input argument and as the output argument.
      eval(['handles = Alg',handles.(AlgName),'(handles);'])
    catch
      %%% copied from gui's errorfunction
      Error = lasterr;
      %%% If an error occurred in an image analysis module, the error message
      %%% should begin with "Error using ==> Alg", which will be recognized here.
      if strncmp(Error,'Error using ==> Alg', 19) == 1
        ErrorExplanation = ['There was a problem running the analysis module number ',AlgNumberAsString, '.', Error];
        %%% The following are errors that may have occured within the analyze all
        %%% images callback itself.
      elseif isempty(strfind(Error,'bad magic')) == 0
        ErrorExplanation = 'There was a problem running the image analysis. It seems likely that there are files in your image directory that are not images or are not the image format that you indicated. Probably the data for the image sets up to the one which generated this error are OK in the output file.';
      else
        ErrorExplanation = ['There was a problem running the image analysis. Sorry, it is unclear what the problem is. It would be wise to close the entire CellProfiler program in case something strange has happened to the settings. The output file may be unreliable as well. Matlab says the error is: ', Error];
      end
      handles_results.CellProfilerError = ErrorExplanation %#ok We want to ignore MLint error checking for this line.
      %%% Breaks out of the image analysis loop.
      break
    end % Goes with try/catch.
  end % Goes with the "if" of the first algorithm running.
end

%%% The only thing read back by the central machine is dM fields.
%%% Copy those to the (empty) results structure
Fields = fieldnames(handles);
mFields = (strncmp(Fields,'dM',2) | strncmp(Fields,'dOTFilename',11));
MeasurementFields = Fields(mFields);

handles_results.setbeinganalyzed = setbeinganalyzed;
for FieldIndex = 1:length(MeasurementFields),
  handles_results.(cell2mat(MeasurementFields(FieldIndex))){setbeinganalyzed} = ...
      handles.(cell2mat(MeasurementFields(FieldIndex))){setbeinganalyzed};
end

disp(['done with set ' int2str(setbeinganalyzed)]);



