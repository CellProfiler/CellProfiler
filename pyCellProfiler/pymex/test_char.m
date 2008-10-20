py_exec('import sys')
py_exec('sys.path.append(".")')
x = 'Fooø';
py_funcall('tests', 'test_char', x); disp(x);
