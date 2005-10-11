function [Success] = CompileSubfunctionOnCluster(FullPathofSubfunction, SubfunctionName)

try
    cd(FullPathofSubfunction)
    a = 'test'
    mex([SubfunctionName,'.cpp'])
    Success= 'The subfunction is compiled';
catch
    Success = 'There was an error compiling the subfunction';
    
end