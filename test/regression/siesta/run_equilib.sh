### Tests whether a provided model reproduces the correct equilibrium geometry 
### for a water monomer. Printed forces should be close to 0 eV/Ang.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir workdir 

cp -r * workdir/

cd workdir
if [ -z "$1" ]
  then
    echo "NeuralXC ../model.jit" >> equilib.fdf 
  else
    echo "NeuralXC ../${1}" >> equilib.fdf 
fi
mpirun -np 8 siesta < equilib.fdf > mpi.out 

cp mpi.out ../
cd ..
rm workdir/*
rmdir workdir 

echo "============ Computed forces (MPI) ==========="

grep "siesta: Atomic forces" mpi.out -A 3 | tail -n 4

rm mpi.out
