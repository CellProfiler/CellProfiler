function im = dibread(filename, width, height, channels, depth)

fid = fopen(filename, 'r');
if (fid == -1),
  error(['DIBREAD could not open ' filename]);
end

ignore = fread(fid, 52, 'uchar');

im = zeros(height,width,channels);

for c=1:channels,
  [data, count] = fread(fid, width * height, 'uint16', 0, 'l');
  if count < (width * height),
    fclose(fid);
    error(['end-of-file encountered in DIBREAD while reading ' filename]);
  end
  im(:,:,c) = reshape(data, [width height])' / (2^depth - 1);
end

fclose(fid);
