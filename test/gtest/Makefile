ARCH_MAKE_DEFAULT=../../arch.make
ARCH_MAKE?=$(ARCH_MAKE_DEFAULT)
include $(ARCH_MAKE)

default: cuda.o grid.o atomic.o run_all.o
	$(CXX) -L/usr/local/lib/gtest ../../libnxc.so *.o -o run_all.x -lgtest -pthread
	./run_all.x
	@make clean

.cpp.o:
	$(CXX) -I/usr/local/include/ -I$(TORCH) -I../../src/ $< -c

clean:
	@rm *.o *.x
