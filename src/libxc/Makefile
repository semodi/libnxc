LIBXCDIR=/home/sebastian/Research/libxc-4.3.4/src
CFILES=$(wildcard *.c)
TARGETS=$(patsubst %.c, $(LIBXCDIR)/%.c , $(CFILES))
TARGETSRM=$(patsubst %.c, %.crm , $(CFILES))

default: $(TARGETS)

$(LIBXCDIR)/%.c : %.c
	ln -s $(CURDIR)/$< $(LIBXCDIR)/$*.c

.PHONY: clean
clean: $(TARGETSRM)
%.crm :
	unlink $(LIBXCDIR)/$*.c
