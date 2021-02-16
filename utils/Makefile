.SUFFIXES:
.SUFFIXES: .o .F90 .f90 .F .f .a .cpp
VPATH=../src


# Compile with MPI support (only needed for NeuralXC models)	
ifneq ($(MPI),)
MPIFLAG=-DMPI
else
MPIFLAG=
endif

ARCH_MAKE_DEFAULT=../arch.make
ARCH_MAKE?=$(ARCH_MAKE_DEFAULT)
include $(ARCH_MAKE)


CXXFLAGS += -DNXC_DEFAULTPATH=${CURDIR}/../models/
LDFLAGS += $(TORCH_LD)
# If LIBXCDIR is set in arch.make, create symbolic links in libxc source directory
ifneq ($(LIBXCDIR),)
all: libxc libnxc
else
all: libnxc
endif


.PHONY: libxc libnxc clean
libxc:
	cd ../src/libxc && $(MAKE) 

libnxc: defaults.o gridfunc.o nxc_func.o atomicfunc.o nxc.o nxc_f90.o nxc_c.o 
	$(FC) *.o $(LDFLAGS) -o libnxc.so
	@mv libnxc.so ../libnxc.so


.cpp.o:
	$(CXX) -Ilibnxc -c $(CXXFLAGS) $(INCFLAGS) $(MPIFLAG) $< 
clean:
	@rm -f module_built
	@rm -f *.o
	@rm -f libnxc.so
	cd ../src/libxc && $(MAKE) clean



