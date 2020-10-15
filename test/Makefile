.SUFFIXES: .o .F90 .f90 .F .f .a 

default: test

ARCH_MAKE_DEFAULT=../arch.make
ARCH_MAKE?=$(ARCH_MAKE_DEFAULT)
include $(ARCH_MAKE)
#
#

test: test_fort test_c clean

test_fort: test_fort.o
	$(FC) test_fort.o ../libnxc.so -o test_fort
	@echo "Importing models in fortran"
	@./test_fort

test_c: test_c.o
	$(CXX) test_c.o ../libnxc.so -o test_c
	@echo "Importing models in C++"
	@./test_c

test_fort.o: 
	$(FC) -c test_fort.f

test_c.o: 
	$(CXX) -c test_c.cpp -I$(TORCH) -I../src/

clean:
	@rm -f *.o
	@rm -f test_fort
	@rm -f test_c



