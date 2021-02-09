CXX = mpic++
CC = gcc
FC = mpif90 --free-form
FC_SERIAL = gfortran --free-form

LDFLAGS =-L/usr/lib/ -lgfortran
LIBS= $(LD_FLAGS) -lstdc++

TORCH=/home/sebastian/lib/libtorch

GTEST= -L/usr/local/lib/gtest -lgtest

LIBXC_INCLUDE=-I/opt/etsf/include/
LIBXC_LD=-L/opt/etsf/lib/ -lxc
LIBXCDIR=/home/sebastian/Research/libxc-4.3.4/src
