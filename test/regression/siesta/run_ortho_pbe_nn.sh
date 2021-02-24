### Compares output of calculation (on water monomer) to reference results

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir workdir 

cp * workdir/

cd workdir
siesta < ortho.fdf > pbe.out 
echo "NeuralXC GGA_XC_PBE" >> ortho.fdf 
siesta < ortho.fdf > serial.out 

cp serial.out ../
cp pbe.out ../
cd ..
rm workdir/* 
rmdir workdir 

serial=$(grep 'siesta: *. Total' serial.out | awk '{printf "%.7f", $4}')
expected=$(grep 'siesta: *. Total' pbe.out | awk '{printf "%.7f", $4}')

echo "${serial} ${expected}" | awk '{printf("Delta E (serial - expected): %.7f\n", $1-$2)}'
echo "${serial} ${expected}" | awk '{printf("%s\n", ($1-$2)*($1-$2) < 0.001*0.001 ? "\033[92m Energies match \033[0m": " \033[91m Energies do not match \033[0m")}'

echo "============ Expected forces ==========="
grep "Atomic forces" pbe.out -A 3 | tail -n 4

echo "============ Computed forces (serial) ==========="
grep "siesta: Atomic forces" serial.out -A 3 | tail -n 4

echo ""
echo "============ Expected stress ==========="
grep 'Stress tensor' pbe.out -A 3


echo "============ Computed stress (serial) ==========="
grep 'Stress tensor' serial.out -A 3

rm serial.out
rm pbe.out
