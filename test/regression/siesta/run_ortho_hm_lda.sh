### Compares output of calculation (on water monomer) to reference results

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir workdir 

cp * workdir/

cd workdir
echo "NeuralXC LDA_HM" >> ortho.fdf 
siesta < ortho.fdf > serial.out 
mpirun -np 8 siesta < ortho.fdf > mpi.out 

cp serial.out ../
cp mpi.out ../
cd ..
rm workdir/* 
rmdir workdir 

expected=$(grep ' *. Total' expected_output_hm_lda | awk '{printf "%.7f", $3}')
serial=$(grep 'siesta: *. Total' serial.out | awk '{printf "%.7f", $4}')
mpi=$(grep 'siesta: *. Total' mpi.out | awk '{printf "%.7f", $4}')

echo "${serial} ${expected}" | awk '{printf("Delta E (serial - expected): %.7f\n", $1-$2)}'
echo "${serial} ${expected}" | awk '{printf("%s\n", $1-$2 < 0.001 ? "\033[92m Energies match \033[0m": " \033[91m Energies do not match \033[0m")}'

echo "${serial} ${mpi}" | awk '{printf("Delta E (serial - expected): %.7f\n", $1-$2)}'
echo "${serial} ${mpi}" | awk '{printf("%s\n", $1-$2 < 0.001 ? "\033[92m Energies match \033[0m": " \033[91m Energies do not match \033[0m")}'

echo "============ Expected forces ==========="
grep "Atomic forces" expected_output_hm_lda -A 3 | tail -n 4

echo "============ Computed forces (serial) ==========="
grep "siesta: Atomic forces" serial.out -A 3 | tail -n 4


echo "============ Computed forces (MPI) ==========="
grep "siesta: Atomic forces" mpi.out -A 3 | tail -n 4

echo ""
echo "============ Expected stress ==========="
grep 'Stress tensor' expected_output_hm_lda -A 3


echo "============ Computed stress (serial) ==========="
grep 'Stress tensor' serial.out -A 3


echo "============ Computed stress (MPI) ==========="
grep 'Stress tensor' mpi.out -A 3

rm serial.out
rm mpi.out
