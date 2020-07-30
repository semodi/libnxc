.SUFFIXES:
.SUFFIXES: .o .F90 .f90 .F .f .a .cpp
VPATH=../src
#
#

CXXFLAGS=-DAT_PARALLEL_OPENMP=1 -isystem $(TORCH)/include -isystem $(TORCH)/include/torch/csrc/api/include  -DBOOST_MATH_DISABLE_FLOAT128 -fPIC   -D_GLIBCXX_USE_CXX11_ABI=1 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-write-strings -Wno-unknown-pragmas -Wno-missing-braces -fopenmp -std=gnu++17 -I/usr/include
MPIFLAG=-DMPI

ifneq ($(MPI),)
CPPFLAGS=$(MPIFLAG)
default: libnxc_mpi.so
else
default: libnxc_serial.so
endif

ARCH_MAKE_DEFAULT=../arch.make
ARCH_MAKE?=$(ARCH_MAKE_DEFAULT)
include $(ARCH_MAKE)
#
#
libnxc_serial.so: atomicfunc.o nxc.o nxc_f90.o
	$(FC) *.o -DBOOST_MATH_DISABLE_FLOAT128 -fPIC  -Wl,-rpath,/lib -L/lib  -Wl,-rpath,$(TORCH)/lib/ $(TORCH)/lib/libtorch.so $(TORCH)/lib/libc10.so -Wl,--no-as-needed,$(TORCH)/lib/libtorch_cpu.so -Wl,--as-needed $(TORCH)/lib/libc10.so -lpthread -Wl,--no-as-needed,$(TORCH)/lib/libtorch.so -Wl,--as-needed -lgfortran -lquadmath -lstdc++ -shared -o libnxc.so
	@mv libnxc.so ../libnxc_serial.so

libnxc_mpi.so: atomicfunc.o nxc.o nxc_f90.o
	$(FC) *.o -DBOOST_MATH_DISABLE_FLOAT128 -fPIC  -Wl,-rpath,/lib -L/lib  -Wl,-rpath,$(TORCH)/lib/ $(TORCH)/lib/libtorch.so $(TORCH)/lib/libc10.so -Wl,--no-as-needed,$(TORCH)/lib/libtorch_cpu.so -Wl,--as-needed $(TORCH)/lib/libc10.so -lpthread -Wl,--no-as-needed,$(TORCH)/lib/libtorch.so -Wl,--as-needed -lgfortran -lquadmath -lstdc++ -shared -o libnxc.so
	@mv libnxc.so ../libnxc_mpi.so
#
#
#
.cpp.o:
	$(CXX) -Ilibnxc -c $(CXXFLAGS) $(INCFLAGS) $(CPPFLAGS) $< 
clean:
	@rm -f module_built
	@rm -f *.o
	@rm -f libnxc.so



