function [Success] = CPcompilesubfunction(FullPathofSubfunction, SubfunctionName)

try
    cd(FullPathofSubfunction)
    a = 'test'
    mex([SubfunctionName,'.cpp'])
    Success= 'The subfunction is compiled';
catch
    Success = 'There was an error compiling the subfunction';
    
end

%%% Here is what you would type at the command line to compile a subfunction:
%%%