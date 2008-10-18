def test_numbers(a):
    print a[0][0]

def test_struct(a):
    print a

def test_char(a):
    print a

def test_cell(a):
    print "foo"
    print repr(a)
    print "foo"
    print dict(a)
    print "foo"
    print a
    print "foo"

def test_echo(a):
    print a
    print len(a)
    print type(a)

def test_handles(a):
    for k in a['handles'][0,0]:
        print k
