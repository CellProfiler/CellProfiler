function result = CPverLessThan(toolboxstr, verstr)

% This CP function is present only so we can easily replace the
% verlessThan if necessary.  See documentation for warndlg for usage.
    
error(nargchk(2, 2, nargin, 'struct'))
        
if ~ischar(toolboxstr) || ~ischar(verstr)
    error('MATLAB:verLessThan:invalidInput', 'Inputs must be strings.')
end

if ~isdeployed,
    toolboxver = ver(toolboxstr);
    if isempty(toolboxver)
        error('MATLAB:verLessThan:missingToolbox', 'Toolbox ''%s'' not found.', toolboxstr)
    end

    toolboxParts = getParts(toolboxver(1).Version);
else
    toolboxver = version;
    if isempty(toolboxver)
        error('MATLAB:verLessThan:missingToolbox', 'Toolbox ''%s'' not found.', toolboxstr)
    end
    toolboxParts = getParts(toolboxver);
end
verParts = getParts(verstr);

result = (sign(toolboxParts - verParts) * [1; .1; .01]) < 0;

function parts = getParts(V)
    parts = sscanf(V, '%d.%d.%d')';
    if length(parts) < 3
       parts(3) = 0; % zero-fills to 3 elements
    end
