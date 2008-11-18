function labels_out = CPremap_from_example(example_in, example_out, labels_in)
Map = sparse(1:numel(example_in), example_in(:)+1, example_out(:));
LookUpColumn = full(max(Map,[], 1));
LookUpColumn(1)=0;
labels_out = LookUpColumn(labels_in+1);
