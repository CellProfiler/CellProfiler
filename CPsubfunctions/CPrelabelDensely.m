% Renumbers the objects in a label matrix so that they are numbered
% consecutively.
function labels = CPrelabelDensely(labels)
props = regionprops(labels, {'Area', 'PixelIdxList'});
present = find([props.Area] > 0);
missing = find([props.Area] == 0);
for i=1:length(missing)
  object_to_relabel = present(length(present) - i + 1);
  if object_to_relabel > missing(i)
    labels(props(object_to_relabel).PixelIdxList) = missing(i);
  end
end
