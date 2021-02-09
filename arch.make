CXX = mpic++
CC = gcc
FC = mpif90 --free-form
FC_SERIAL = gfortran --free-form
LDFLAGS =-L/usr/lib/ -lgfortran
TORCH=/home/sebastian/lib/libtorch
LIBS= $(LD_FLAGS) -lstdc++
