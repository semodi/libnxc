CXX = mpic++
CC = gcc
FC = mpif90
FC_SERIAL = gfortran

CPPFLAGS = -DMPI 

LDFLAGS =-L/usr/lib/ -lgfortran

TORCH=/home/sebastian/lib/libtorch

LIBS= $(LD_FLAGS) -lstdc++


