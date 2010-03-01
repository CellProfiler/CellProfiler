py_exec('import sys')
py_exec('sys.path.append(".")')
x(1) = struct('a', 1, 'b', 2);
x(2) = struct('a', 3, 'b', 4);
py_funcall('tests', 'test_struct', x); disp(x);
