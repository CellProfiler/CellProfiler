function ScaledImage = CPimread_flex(imname, idx)
%%% Read a .flex file, with possible scaling information.  No
%%% documentation is available, as far as I know.  Most of what is
%%% below is based on experiments, looking at the plaintext XML in the
%%% .flex files, and by reference to FlexReader.java in the LOCI
%%% Bio-Formats project.  


% First load the image...
RawImage = imread(imname, idx);

% Then go back and try to get the scaling factors...
try
    % get the file contents
    fid = fopen(imname, 'r');
    FileAsString = fread(fid, inf, 'uint8=>char')';
    fclose(fid);

    % Extract the Array information
    strstart = findstr('<Arrays>', FileAsString);
    strend = findstr('</Arrays>', FileAsString);
    ArrayString = FileAsString(strstart(1):strend(1));

    % Find the Factors
    factor_locations = findstr('Factor="', ArrayString);
    
    % determine maximum factor value, as it decides the number of bits to convert to, below.
    for i = 1:length(factor_locations),
        % get the ith factor string, and extract the value
        IdxFactorStringStart = ArrayString(factor_locations(i) + length('Factor="'):end);
        strend = findstr(IdxFactorStringStart, '"') - 1;
        IdxFactorString = IdxFactorStringStart(1:strend);
        ScalingFactors(i) = str2double(IdxFactorString);
    end

    %%% The logic here mirrors that in FlexReader.java, part of the LOCI Bio-Formats package
    if max(ScalingFactors) > 256,
        % upgrade to 32 bits
        ScaledImage = uint32(RawImage) * ScalingFactors(idx);
    elseif max(ScalingFactors) > 1,
        % upgrade to 16 bits
        ScaledImage = uint16(RawImage) * ScalingFactors(idx);
    else
        if isa(RawImage, 'uint8'),
            % FlexReader.java leaves this as 8 bits, but that seems like
            % it could drop a lot of precision.  Instead, we'll upgrade to
            % 16 bits and multiply by 256, to give some room at the low
            % end of the precision scale.
            ScaledImage = (uint16(RawImage) * 256) * ScalingFactors(idx);
        else
            % We already have sufficient precision
            ScaledImage = RawImage * ScalingFactors(idx);
        end
    end

catch
    % default is no scaling (for non-flex tiffs)
    ScaledImage = RawImage;
end
