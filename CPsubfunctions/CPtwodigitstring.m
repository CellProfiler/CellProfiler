function twodigit = CPtwodigitstring(val)
%TwoDigitString is a function like num2str(int) but it returns a two digit
%representation of a string for our purposes.

% $Revision$

if ((val > 99) || (val < 0)),
    error(['TwoDigitString: Can''t convert ' num2str(val) ' to a 2 digit number']);
end
twodigit = sprintf('%02d', val);