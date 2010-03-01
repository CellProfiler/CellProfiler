py_exec('import sys')
py_exec('sys.path.append(".")')

x{1,1} = 42;
x{1,2} = 'Foo';
x{2,1} = 23;
%x{2,2} = 11;
x = {'LoadImages', 'DICTransform'};
disp(x)
py_funcall('tests', 'test_cell', x);
