### Tests model for non-orthorhombic unitcells, by calculating 
### Hexagonal ice in a primitive and extended unitcell. Energies and Stress should match.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir workdir 

cp * workdir/

cd workdir
echo "NeuralXC ../model.jit" >> ice_ortho.fdf 
echo "NeuralXC ../model.jit" >> ice_hex.fdf 
mpirun -np 8 siesta < ice_ortho.fdf > ice_ortho.out 
mpirun -np 8 siesta < ice_hex.fdf > ice_hex.out 

cp ice_ortho.out ../
cp ice_hex.out ../
cd ..
rm workdir/*
rmdir workdir 

ortho=$(grep 'siesta: *. Total' ice_ortho.out | awk '{printf "%.5f", $4}')
hex=$(grep 'siesta: *. Total' ice_hex.out | awk '{printf "%.5f", 2*$4}')
echo "${ortho} ${hex}"  | awk '{printf("Delta E: %.5f\n", $1-$2)}'
echo "${ortho} ${hex}"  | awk '{printf("%s\n", ($1-$2)*($1-$2) < 0.003*0.003 ? "\033[92m Energies match \033[0m": " \033[91m Energies do not match \033[0m")}'
echo ""

echo "============ Computed stress (ortho) ==========="
grep 'Stress tensor' ice_ortho.out -A 3

echo "============ Computed stress (hex) ==========="
grep 'Stress tensor' ice_hex.out -A 3


rm ice_ortho.out
rm ice_hex.out
