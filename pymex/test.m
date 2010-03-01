py_exec('import sys')
py_exec('sys.path.append(".")')
py_funcall('test_module', 'foo', [1,2;3,4])
py_funcall('test_module', 'foo', [5,6,7,8])
