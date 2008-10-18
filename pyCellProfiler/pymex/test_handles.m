py_exec('import sys')
py_exec('sys.path.append(".")')

handles = load('~/research/cellprofiler/tmp/DefaultOUT__36.mat');

mh = handles.handles.Current.ModulesHelp;
py_funcall('tests', 'test_echo', mh(53));

