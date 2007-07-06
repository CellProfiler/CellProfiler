#DEBUG=-O3 -DNDEBUG
DEBUG=-g
CFLAGS = -Wall -pedantic -ansi -I.. $(DEBUG) --std=c99

PRODUCTS = libvlcutils.a test_bplus test_readline

all: $(PRODUCTS)

libvlcutils.a: error.o heap.o intset.o bplus.o plist.o mem.o intlist.o \
		readline.o bitmap.o pgm.o io.o progress.o
	ar rc $@ $+
	ranlib $@

error.o: error.c error.h
heap.o: heap.c heap.h
intset.o: intset.c intset.h
bplus.o: bplus.c bplus.h
plist.o: plist.c plist.h
mem.o: mem.c mem.h
intlist.o: intlist.c intlist.h
obintlist.o: obintlist.c obintlist.h
readline.o: readline.c readline.h
bitmap.o: bitmap.c bitmap.h
pgm.o: pgm.c pgm.h
io.o: io.c io.h
progress.o: progress.c progress.h

test_bplus.o: bplus.c bplus.h

test_bplus: test_bplus.o error.o intset.o
	$(CC) -o $@ $+

test_readline: test_readline.o readline.o mem.o error.o
	$(CC) $(CFLAGS) -o $@ $+

clean:
	rm -f $(PRODUCTS) *.o
