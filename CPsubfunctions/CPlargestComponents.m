function components_labels = CPlargestComponents(labels)
components = bwlabel(labels > 0);
components = regionprops(components, {'Area', 'PixelIdxList'});
n = length(components);
for i=1:n
  components(i).label = labels(components(i).PixelIdxList(1));
end
largest = zeros(max(labels(:)),2);
for i=1:n
  label = components(i).label;
  area = components(i).Area;
  if largest(label,2) == 0 || area > largest(label,1)
    largest(label,:) = [area i];
  end
end
components_labels = labels;
for i=1:n
  label = components(i).label;
  if largest(label,2) ~= i
    components_labels(components(i).PixelIdxList) = 0;
  end
end
