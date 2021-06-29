ARCH_MAKE_DEFAULT=../../arch.make
ARCH_MAKE?=$(ARCH_MAKE_DEFAULT)
include $(ARCH_MAKE)
CFILES=$(wildcard *.c)
TARGETS=$(patsubst %.c, $(LIBXCDIR)/%.c , $(CFILES))
TARGETSRM=$(patsubst %.c, %.crm , $(CFILES))

default: $(TARGETS)

$(LIBXCDIR)/%.c : %.c
	ln -s $(CURDIR)/$< $(LIBXCDIR)/$*.c
	cp $(CURDIR)/../../utils/libxc/Makefile.am $(LIBXCDIR)
	cp $(CURDIR)/../../utils/libxc/CMakeLists.txt $(LIBXCDIR)

.PHONY: clean
clean: $(TARGETSRM)
%.crm :
	unlink $(LIBXCDIR)/$*.c
