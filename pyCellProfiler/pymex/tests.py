def test_numbers(a):
    print a[0][0]

def test_struct(a):
    print repr(a)

def test_char(a):
    print repr(a)

def test_funky_encoding(a):
    print a.encode('latin1').decode('cp1252')

def test_cell(a):
    print "foo"
    print repr(a)
    print "foo"
    print dict(a)
    print "foo"
    print a
    print "foo"

def test_echo(a):
    print type(a)
    print len(a)
    print a.dtype
    print repr(a)
    print a

def test_handles(a):
    #print a['handles']
    for k in a['handles'][0,0].dtype.names:
        print k

def test_handles_handles(a):
    for k in a[0,0].dtype.names:
        print k
        print a[k]
