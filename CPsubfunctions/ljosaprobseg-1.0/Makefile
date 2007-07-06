#DEBUG = -g
DEBUG = -O3 -DNDEBUG
INCLUDES = -I.
CFLAGS = $(INCLUDES) -Wall -pedantic -ansi $(DEBUG)
MEXEXT := $(shell mexext)
PACKAGE := $(shell grep PACKAGE config.h | sed 's/.*"\(.*\)".*/\1/')
VERSION := $(shell grep VERSION config.h | sed 's/.*"\(.*\)".*/\1/')

PROGRAMS = ljosaprobsegsim ljosaprobseg.$(MEXEXT)

all: $(PROGRAMS)

ljosaprobsegsim: rwalk.o rwalk1.o rwalkpthread.o seed.o vlcutils/libvlcutils.a
	$(CC) -o $@ $+ -Lvlcutils -lvlcutils -lm -lpthread $(DEBUG)

ljosaprobseg.$(MEXEXT): rwalkmex.c rwalk1.o rwalkpthread.o seed.o
	mex -output $@ $(INCLUDES) $+ -Lvlcutils -lvlcutils -lpthread -O

vlcutils/libvlcutils.a:
	make -C vlcutils

.c.o:
	$(CC) -c $< $(CFLAGS)

clean:
	@rm -f $(PROGRAMS) *.o $(PACKAGE)-$(VERSION).tar.gz .gdb_history
	@rm -rf $(PACKAGE)-$(VERSION)/
	@make -C vlcutils clean

dist: clean
	mkdir $(PACKAGE)-$(VERSION)
	tar cf - --exclude $(PACKAGE)-$(VERSION) --dereference . | tar xf - -C $(PACKAGE)-$(VERSION)
	tar cvzf $(PACKAGE)-$(VERSION).tar.gz --exclude '*~' --exclude '.??*' --exclude CVS $(PACKAGE)-$(VERSION)
	rm -rf $(PACKAGE)-$(VERSION)/
