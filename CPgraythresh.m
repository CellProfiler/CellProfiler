function [level] = CPgraythresh(im)

lo = min(im(:));
hi = max(im(:));
if (lo == hi),
    [level] = graythresh(im);
    return;
end

[level] = graythresh((im - lo) / (hi - lo));
level = level * (hi - lo) + lo;
