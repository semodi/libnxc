ARCH_MAKE_DEFAULT=../../arch.make
ARCH_MAKE?=$(ARCH_MAKE_DEFAULT)
include $(ARCH_MAKE)

ifneq ($(CUDA),)
CPPFLAGS+=-DCUDA
else
endif

ifneq ($(LIBXC),)
CPPFLAGS+=-DLIBXC -I/opt/etsf/include/
LDFLAGS+=-L/opt/etsf/lib/ -lxc -lm
else
endif



default: libxc.o cuda.o grid.o atomic.o run_all.o
	$(CXX) -L/usr/local/lib/gtest ../../libnxc.so *.o -o run_all.x -lgtest -pthread $(LDFLAGS)
	./run_all.x
	@make clean

.cpp.o:
	$(CXX) $(CPPFLAGS) -I/usr/local/include/ -I$(TORCH) -I../../src/ $< -c

clean:
	@rm *.o *.x
