
CC = gcc
CFLAGS=-I/Library/Frameworks/Python.framework/Versions/2.5/include/python2.5 -I/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/numpy/core/include -g
#LDFLAGS=-L/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/config `python -c "import distutils.sysconfig; print distutils.sysconfig.get_config_var('LINKFORSHARED')"`
LDFLAGS=-L/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/config

PROGRAMS = py_exec.mexmaci py_funcall.mexmaci simple_embed \
	minimal_main embed_mex.mexmaci minimal_mex.mexmaci

all: $(PROGRAMS)

py_exec.mexmaci: py_exec.c pymex.c
	mex $(CFLAGS) $+ $(LDFLAGS) -lpython2.5

py_funcall.mexmaci: py_funcall.c pymex.c
	mex $(CFLAGS) $+ $(LDFLAGS) -lpython2.5

simple_embed: simple_embed.c
	$(CC) $(CFLAGS) -o $@ $+ $(LDFLAGS) -lpython2.5

minimal_main: minimal_main.c
	$(CC) -g -O0 $(CFLAGS) -o $@ $+ $(LDFLAGS) -lpython2.5

embed_mex.mexmaci: embed_mex.c pymex.c
	mex $(CFLAGS) $+ $(LDFLAGS) -lpython2.5

minimal_mex.mexmaci: minimal_mex.c
	mex $(CFLAGS) $+ $(LDFLAGS) -lpython2.5

test_pymex: test_pymex.o pymex_py.o
	$(CC) -o $@ $+ $(LDFLAGS) -lpython2.5

clean:
	rm -f $(PROGRAMS) *.o