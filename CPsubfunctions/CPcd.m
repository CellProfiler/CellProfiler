function CurrentDir = CPcd(NewDir)
%%% This function will check to make sure the directory specified exist
%%% before performing the cd function.
if nargin == 0
    if isdir(cd)
        CurrentDir = cd;
    else
        CPwarndlg('This directory no longer exist! This function will default to the Matlab root directory.');
        CurrentDir = matlabroot;
    end
elseif nargin == 1
    if isdir(NewDir)
        CurrentDir = cd(NewDir);
    else
        CPwarndlg('This directory no longer exist! This function will default to the Matlab root directory.');
        CurrentDir = matlabroot;
    end
end