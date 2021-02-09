ARCH_MAKE_DEFAULT=../../arch.make
ARCH_MAKE?=$(ARCH_MAKE_DEFAULT)
include $(ARCH_MAKE)

ifneq ($(CUDA),)
CPPFLAGS+=-DCUDA
else
endif

ifneq ($(LIBXC_INCLUDE),)
CPPFLAGS+=-DLIBXC $(LIBXC_INCLUDE)
LDFLAGS+=$(LIBXC_LD) -lm
else
endif

default: libxc.o cuda.o grid.o atomic.o run_all.o
	$(CXX)  ../../libnxc.so *.o -o run_all.x -pthread $(LDFLAGS) $(GTEST)
	./run_all.x
	@make clean

.cpp.o:
	$(CXX) $(CPPFLAGS) -I/usr/local/include/ -I$(TORCH) -I../../src/ $< -c

clean:
	@rm *.o *.x
