function [level, em] = CPgraythresh(im)

lo = min(im(:));
hi = max(im(:));
if (lo == hi),
    [level, em] = graythresh(im);
    return;
end

[level, em] = graythresh((im - lo) / (hi - lo));
level = level * (hi - lo) + lo;
