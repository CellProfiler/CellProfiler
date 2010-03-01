py_exec('import sys')
py_exec('sys.path.append(".")')
x = int8(-128); py_funcall('tests', 'test_numbers', x); disp(x);
x = int8(127);  py_funcall('tests', 'test_numbers', x); disp(x); 
x = uint8(255); py_funcall('tests', 'test_numbers', x); disp(x); 
x = int16(-32768); py_funcall('tests', 'test_numbers', x); disp(x); 
x = int16(32767); py_funcall('tests', 'test_numbers', x); disp(x); 
x = uint16(65535); py_funcall('tests', 'test_numbers', x); disp(x); 
x = int32(-2147483648); py_funcall('tests', 'test_numbers', x); disp(x); 
x = int32(2147483647); py_funcall('tests', 'test_numbers', x); disp(x); 
x = uint32(4294967295); py_funcall('tests', 'test_numbers', x); disp(x); 
x = int64(-9223372036854775808); py_funcall('tests', 'test_numbers', x); disp(x); 
x = int64(9223372036854775807); py_funcall('tests', 'test_numbers', x); disp(x); 
x = uint64(18446744073709551615); py_funcall('tests', 'test_numbers', x); disp(x); 

x = 2 + j; py_funcall('tests', 'test_numbers', x); disp(x);
x = single(2) + j; py_funcall('tests', 'test_numbers', x); disp(x);

x = true; py_funcall('tests', 'test_numbers', x); disp(x); 
x = false; py_funcall('tests', 'test_numbers', x); disp(x); 
