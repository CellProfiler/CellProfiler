py_exec('import sys')
py_exec('sys.path.append(".")')

handles = load('~/research/cellprofiler/tmp/DefaultOUT__36.mat');
py_funcall('tests', 'test_handles', handles);
