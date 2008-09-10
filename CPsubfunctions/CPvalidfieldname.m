function CPvalidfieldname(fieldname)

% Throw an error if the field name does not start with
% an alphabetic character, if it contains characters other
% than alphanumerics and underbar or if it is more than 63
% characters long.

% $Revision$

if length(fieldname) > namelengthmax
    error(['The field name, "',fieldname,'", is more than ',num2str(namelengthmax),' characters long.']);
end
if isempty(regexp(fieldname,'^[A-Za-z]', 'once' ))
    error(['The field name, "',fieldname,'", does not start with an alphabetic character.']);
end
if isempty(regexp(fieldname,'^[A-Za-z][A-Za-z0-9_]{0,62}$', 'once' ))
    error(['The field name, "',fieldname,'", contains characters other than alphanumerics and "_"']);
end
end