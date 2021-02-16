CXX = mpic++
CC = gcc
FC = mpif90 --free-form
FC_SERIAL = gfortran --free-form

LDFLAGS =-L/usr/lib/ -lgfortran
LIBS= $(LD_FLAGS) -lstdc++

# Libtorch location (required)
TORCH=/home/sebastian/lib/libtorch

# Google test location (optional)
GTEST= -L/usr/local/lib/gtest -lgtest

# Libxc (optional)
LIBXC_INCLUDE=-I/opt/etsf/include/
LIBXC_LD=-L/opt/etsf/lib/ -lxc
LIBXCDIR=/home/sebastian/Research/libxc-4.3.4/src


# The following parts should remain largely unchanged 

CXXFLAGS=-DAT_PARALLEL_OPENMP=1 -isystem $(TORCH)/include -isystem $(TORCH)/include/torch/csrc/api/include  -DBOOST_MATH_DISABLE_FLOAT128 -fPIC   -D_GLIBCXX_USE_CXX11_ABI=1 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-write-strings -Wno-unknown-pragmas -Wno-missing-braces -fopenmp -std=gnu++17 -I/usr/include

TORCH_LD= -DBOOST_MATH_DISABLE_FLOAT128 -fPIC  -Wl,-rpath,/lib -L/lib  -Wl,-rpath,$(TORCH)/lib/ $(TORCH)/lib/libtorch.so $(TORCH)/lib/libc10.so -Wl,--no-as-needed,$(TORCH)/lib/libtorch_cpu.so -Wl,--as-needed $(TORCH)/lib/libc10.so -lpthread -Wl,--no-as-needed,$(TORCH)/lib/libtorch.so -Wl,--as-needed -lgfortran -lquadmath -lstdc++ -shared 

