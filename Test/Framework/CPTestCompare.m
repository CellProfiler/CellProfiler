function results=CPTestCompare(DefaultOUT,ExpectedOUT,results)
if ~isfield(DefaultOUT,'handles')
    throw(MException('CPTestCompare:MissingField','Output file does not have a handles structure'));
end
if ~ isfield(ExpectedOUT,'handles')
    throw(MException('CPTestCompare:BadTest','Expected output file does not have a handles structure'));
end
if ~isfield(DefaultOUT.handles,'Measurements')
    throw(MException('CPTestCompare:MissingField','Output file handles do not have a Measurements structure'));
end
if ~isfield(ExpectedOUT.handles,'Measurements')
    throw(MException('CPTestCompare:BadTest','Expected output file handles do not have a Measurements structure'));
end
results = Compare(DefaultOUT.handles.Measurements,...
    ExpectedOUT.handles.Measurements,'handles.Measurements',results);

function results = Compare(d,e,path,results)
fn=fieldnames(e);
for i=1:length(fn)
    fieldname=fn{i};
    if ~isfield(d,fieldname)
        throw(MException('CPTestCompare:MissingField',[path,'.',fieldname,' is missing from the test output']));
    end
    d1=d.(fieldname);
    e1=e.(fieldname);
    results=CompareField(d1,e1,path,fieldname,results);
end

function results=CompareField(d1,e1,path,fieldname,results)
path=[path,'.',fieldname];
if length(d1)~=length(e1)
    throw(MException('CPTestCompare:DifferentLengths',sprintf('# of elements in %s is different: %d != expected(%d)',path,length(d1),length(e1))));
end
if isnumeric(e1)
    if ~isnumeric(d1)
        throw(MException('CPTestCompare:DifferentTypes',sprintf('%s is not numeric',path)));
    end
    SignificantDigits = 3;
    locs=find(...
        (d1 == 0 && e1 ~= 0) ||...
        (d1 ~= 0 && e1 == 0) ||...
        (log10(abs((d1-e1)/d1)) + SignificantDigits > 0 &&...
         log10(abs((d1-e1)/e1)) + SignificantDigits > 0));
     if ~isempty(locs)
         if length(locs)==1
             throw(MException('CPTestCompare:DifferentValues',sprintf('%s differs at element %d: %f != expected (%f)',path,locs,d1,e1)));
         else
             throw(MException('CPTestCompare:DifferentValues',sprintf('%s differs at %d array elements',path,length(locs))));
         end
    end
elseif iscell(e1)
    if isempty(e1)
        return;
    end
    if ischar(e1{1})
        locs=find(~strcmp(d1,e1));
        if ~ isempty(locs)
            if length(locs)==1
                throw(MException('CPTestCompare:DifferentValues',sprintf('%s differs at element %d: %s != expected (%s)',path,locs,d1{locs},e1{locs})));
            else
                throw(MException('CPTestCompare:DifferentValues',sprintf('%s differs at %d array elements',path,length(locs))));
            end
        end
    end
elseif ischar(e1)
    if ~strcmp(d1,e1)
        throw(MException(sprintf('%s differs: %s != expected (%s)',path,d1,e1)));
    end
elseif isstruct(e1)
    results=Compare(d1,e1,path, results);
end
