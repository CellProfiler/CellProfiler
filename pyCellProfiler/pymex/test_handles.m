py_exec('import sys')
py_exec('sys.path.append(".")')

handles = load('~/research/cellprofiler/tmp/DefaultOUT__36.mat');

mh = handles.handles.Current.ModulesHelp;
py_funcall('tests', 'test_funky_encoding', mh(53));

py_funcall('tests', 'test_handles', handles);


