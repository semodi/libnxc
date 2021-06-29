.SUFFIXES: .o .F90 .f90 .F .f .a
.PHONY: test libxc all gtest
default: all

ARCH_MAKE_DEFAULT=../arch.make
ARCH_MAKE?=$(ARCH_MAKE_DEFAULT)
include $(ARCH_MAKE)
#
#

CPPFLAGS+=-DLIBXC $(LIBXC_INCLUDE)
LDFLAGS+=$(LIBXC_LD) -lm

test: test_fort test_c test_c_lda test_c_gga test_fort_gga test_py_gga clean

libxc: t_libxc_scan t_libxc_lda clean 

all:  test libxc t_gtest
gtest: t_gtest


t_gtest: 
ifdef GTEST
	cd gtest && $(MAKE)
else
	@echo "gtest env variable not set, skipping..."
endif 

t_libxc: test_libxc.o 
ifdef LIBXCDIR
	$(CXX) test_libxc.o ../libnxc.so $(LDFLAGS) -o test_libxc
	@echo "Testing libxc"
	@./test_libxc
else
	@echo "not using libxc, skipping..."
endif 

t_libxc_scan: test_libxc_scan.o 
ifdef LIBXCDIR
	$(CXX) test_libxc_scan.o ../libnxc.so $(LDFLAGS) -o test_libxc_scan
	@echo "Testing libxc"
	@./test_libxc_scan
endif

t_libxc_lda: test_libxc_lda.o 
ifdef LIBXCDIR
	$(CXX) test_libxc_lda.o ../libnxc.so $(LDFLAGS) -o test_libxc_lda
	@echo "Testing libxc"
	@./test_libxc_lda
endif

test_fort: test_fort.o
	$(FC) test_fort.o ../libnxc.so -o test_fort
	@echo "Importing models in fortran"
	@./test_fort

test_fort_gga: test_fort_gga.o
	$(FC) test_fort_gga.o ../libnxc.so -o test_fort_gga
	@echo "Testing PBE model in fortran"
	@./test_fort_gga

test_c: test_c.o
	$(CXX) test_c.o ../libnxc.so -o test_c
	@echo "Importing models in C++"
	@./test_c

test_c_gga: test_c_gga.o
	$(CXX) test_c_gga.o ../libnxc.so -o test_c_gga
	@echo "Testing PBE model in C++"
	@./test_c_gga

test_c_lda: test_c_lda.o
	$(CXX) test_c_lda.o ../libnxc.so -o test_c_lda
	@echo "Testing LDA_X model in C++"
	@./test_c_lda

test_py_gga: test_c_gga.o
	@echo "Testing PBE model in Python"
	@python  test_py_gga.py
	
test_fort.o:
	$(FC) -c test_fort.f

test_fort_gga.o:
	$(FC) -c test_fort_gga.f


.cpp.o:
	$(CXX) $(CPPFLAGS) -I/usr/local/include/ -I$(TORCH) -I../src/ $< -c

clean:
	@rm -f *.o
	@rm -f test_fort
	@rm -f test_c
	@rm -f test_c_gga
	@rm -f test_c_lda
	@rm -f test_fort_gga
	@rm -f test_libxc_scan
	@rm -f test_libxc_lda
