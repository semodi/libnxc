
ARCH_MAKE_DEFAULT=../../arch.make
ARCH_MAKE?=$(ARCH_MAKE_DEFAULT)
include $(ARCH_MAKE)

default: atomicfunc.o run_all.o
	g++ -L/usr/local/lib/gtest ../../libnxc.so run_all.o -o run_all.x -lgtest -pthread 
	./run_all.x
	@make clean

atomicfunc.o:
	g++ -I/usr/local/include/ -I$(TORCH) -I../../src/ atomicfunc.cpp -c

run_all.o:
	g++ -I/usr/local/include/ -I$(TORCH) -I../../src/ run_all.cpp -c

clean:
	@rm *.o *.x
