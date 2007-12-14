% Rethrow the error with a more informative message, preserving the
% original stack trace.
function CPerrorImread(ModuleName, n)
ErrorNumber = {'first','second','third','fourth'};
err = lasterror();
err.message = ['Image processing was canceled in the ', ModuleName, ' module because an error occurred when trying to load the ', ErrorNumber{n}, ' set of images. Please check the settings. A common problem is that there are non-image files in the directory you are trying to analyze. Matlab says the problem is: ', err.message];
error(err);
