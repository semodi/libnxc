# Lib**n**xc

Libnxc is a libary to deploy **machine learned** exchange-correlation functionals for density functional theory.

All common functional types (LDA, GGA, metaGGA) as well as NeuralXC type functionals  are supported.

Libnxc is written in C++ and has Fortran bindings. An implementation in Python, `pylibnxc` is also available.

Libnxc is inspired by Libxc, mirroring as closely  as possible its API. In doing so, the integration of Libnxc in electronic structure codes that use Libxc should be straightforward.

Libnxc can utilize multi-processing through MPI and model inference on GPUs through CUDA is supported as well.

## Dependencies

Libnxc depends on PyTorch. The C++/Fortran implementation requires that libtorch is made available at compile time, pylibnxc depends on pytorch.
Both libraries are available for download on the [pytorch website](https://pytorch.org/get-started/locally/).

pylibnxc further depends on numpy, which is a requirement for pytorch and should therefore be installed automatically.

**Optional dependencies:**

For unit testing, pylibnxc currently requires [pyscf](https://sunqm.github.io/pyscf/install.html) which can be obtained through

`pip install pyscf`

To unit test the C++/Fortran implementation [GoogleTest](https://github.com/google/googletest) is required.




## Quickstart

The functionality of Libnxc may be best explained by an example

```c++
#include "nxc.h"

int main()
{
  nxc_func_type p;
  func_param fp;
  int nspin = NXC_UNPOLARIZED;

  double rho[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
  double sigma[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
  double exc[5];

  double vrho[5];
  double vsigma[5];

  nxc_func_init(&p, "GGA_PBE", fp, nspin);

  nxc_gga_exc_vxc(&p, 5, rho, sigma, exc, vrho, vsigma);

}
```

- `nxc_func_type p` stores the functional
- `func_param fp` stores parameters that determine how the functional is evaluated and how Libnxc communicates with the external code. For now, it suffices to know that these parameters are set to sensible default values. We will return to it later.
- `nxc_func_init(&p, "GGA_PBE", fp, nspin)` loads the functional "GGA_PBE", the second argument can either be the name of a shipped functional (see below for a list thereof) or a path to a serialized TorchScript model (at the moment custom models are only available in pylibnxc).
- `nxc_gga_exc_vxc(&p, 5, rho, sigma, exc, vrho, vsigma)` evaluates the functional and stores the energy per unit particle, and the potential terms $\delta E/ \delta \rho(\mathbf r)$ and $\delta E/ \delta \sigma(\mathbf r)$ in `exc`, `vrho` and `vsigma`, respectively.



Running the program above should produce the following output:
```
0.100000   -.459787
0.200000   -.507308
0.300000   -.562453
0.400000   -.611107
0.500000   -.653520
```

The same program can be run from Fortran:

```fortran
PROGRAM test

      implicit none
      integer               :: ierr, i
      character(len=100)    :: nxc_path
      real(8)               :: rho(5), sigma(5), exc(5), vrho(5), vsigma(5)
      ! Initialize grid, basis, etc.
      nxc_path='GGA_PBE'

      rho = (/0.1,0.2,0.3,0.4,0.5/)
      sigma = (/0.2,0.3,0.4,0.5,0.6/)
      call nxc_f90_set_code(0)
      call nxc_f90_func_init(nxc_path, LEN_TRIM(nxc_path), ierr)

      call nxc_f90_gga_exc_vxc(5, rho, sigma, exc, vrho, vsigma)

      do i=1,5
        write(*,"(T1,F8.6,T12,F8.6)") rho(i) , exc(i)
      end do

END PROGRAM test
```

... and Python

```python
from pylibnxc import LibNXCFunctional
import numpy as np


if __name__ == '__main__':

    func = LibNXCFunctional("GGA_PBE")
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sigma = np.array([0.2, 0.3, 0.4, 0.5, 0.6])

    inp = {'rho': rho,
           'sigma': sigma}

    results = func.compute(inp)

    print(np.stack([inp['rho'], np.round(results['zk'],6)],axis=-1))
```

## Using pylibnxc with PySCF

Using pylibnxc in PySCF is as simple as changing two lines of code:

Starting from
```python
from pyscf import gto
from pyscf.dft import RKS

mol = gto.M(atom='H 0 0 0; H 0 0 0.7', basis='6-311G')
mf = RKS(mol)
mf.xc = 'PBE'
mf.kernel()
```
one can use pylibnxc
```python
from pyscf import gto
from pylibnxc.pyscf import RKS

mol = gto.M(atom='H 0 0 0; H 0 0 0.7', basis='6-311G')
mf = RKS(mol, nxc='GGA_PBE', nxc_kind='grid')
mf.kernel()
```
The second version would run a SCF calculation using our machine-learned version of
PBE (see Shipped Models).

For unrestricted Kohn-Sham calculations `pylibnxc.pyscf.UKS` is available as well.
The `nxc` keyword supports mixing of functionals similar to pyscf, e.g.
`nxc ='0.25*HF + 0.75*GGA_X_PBE, GGA_C_PBE'` would correspond to a neural network version
of PBE0.
Currently, mixing of libxc functionals with libnxc functionals is not supported.

## C++ Interface

### Functional parameters
```c++
/**
* @param pos atomic positions
* @param nua number of atoms
* @param cell lattice vectors
* @param grid number of grid points for each lattice vector
* @param isa species index for every atom
* @param symbols ditinct symbols
* @param ns symbols.size()
* @param myBox box in simulation cell (used mainly for MPI decomposition)
* @param edens 0: return total energy, 1: return energy density  (default: 1)
* @param add 0: set return values 1: add return values (default: 1)
* @param cuda 0: use cpu 1: use gpu (default: 0)
*/
struct func_param{
  double * pos; // atomic positions
  int nua; // number of atoms (pos.size())
  double * cell; // lattice vectors
  int * grid; // number of grid points for each LV
  int * isa; // species index for every atom  (relates to symbols array)
  char * symbols; //distinct symbols
  int ns; // symbols.size()
  int * myBox; // box in simulation cell (used mainly for MPI decomposition)
  int edens = defaults->edens;
  int add = defaults->add;
  int cuda = defaults->cuda;
  int gamma = defaults->gamma;
};
```

The struct `func_param` is used to set parameters that relate to the system that is being
simulated as well as the way Libnxc communicates with the electronic structure code.
Parameters `pos` to `myBox` are only relevant for atomic functionals (NeuralXC) and contain information
about the simulation box (unit cell) as well as the atomic positions and species.
`isa` together with `symbols` defines the element of every atom which is relevant if the NeuralXC
functional is species dependent.
Example:
```c++
int * isa = {1,0,0}
char * symbols = {'H','O'}
```
would be interpreted as one oxygen atoms and two hydrogen atoms {'O','H','H'}.

The remaining parameters govern the evaluation of the functional:
- `eden`:
  {`1`: return the energy per unit particle on a grid, (default),
   `0`: only return the total (xc)-energy (This was mainly implemented for performance reasons)}
- `add`:
  {`1`: adds return values for exc and the potential terms to the provided arrays,
   `0`: sets the return values for exc and the potential terms in the provided arrays (default)}
- `cuda`:
  {`1`: model inference on GPUs,
   `0`: model inference on CPUs (default)}
- `gamma`:
  {`1`: for GGAs and higher, the gradient of the electron density is provided,
   `0`: the reduced gradient sigma is provided (default)}

The default values were chosen to closely mirror the functionality of Libxc.
If Libnxc is being used with SIESTA or CP2K, appropriate values can be set with `nxc_set_code`
```c++
const int DEFAULT_CODE=0;
const int SIESTA_GRID_CODE=1;
const int SIESTA_ATOMIC_CODE=2;
const int CP2K_CODE=0;

void nxc_set_code(int code);
```
Note that this function has to be called *before* `nxc_func_init` to have any effect.


### Initializing the functional

The functional can be initialized using `nxc_func_init`
```c++
const int NXC_POLARIZED=2;
const int NXC_UNPOLARIZED=1;
/**
* Initializes functional
*
* @param[out] p loaded functional
* @param[in] model string containing either model path or name
* @param[in] fp functional parameters
* @param[in, optional] nspin spin polarized/unpolarized calcuation (default NXC_UNPOLARIZED)
*/
void nxc_func_init(nxc_func_type* p, std::string model, func_param fp, int nspin=NXC_UNPOLARIZED);
```

### Model evaluation
Depending on which rung the loaded functional resides on one of the following methods can be used
for evaluation:
```c++
/**
* Evaluates the functional on provided density if functional is LDA type. This includes atomic functionals
* that only depend on the local density.
*
* @param[in] p functional to evaluate
* @param[in] np number of grid points (size of rho)
* @param[in] rho electron density
* @param[(in), out] exc energy density. If fp.edens = 0, exc[0] contains energy.
* @param[(in), out] vrho dE/drho
*/
void nxc_lda_exc_vxc(nxc_func_type* p, int np, double rho[], double * exc, double vrho[]);
void nxc_lda_exc_vxc_fs(nxc_func_type* p, int np, double rho[], double * exc, double vrho[],
                        double forces[], double stress[]);
void nxc_gga_exc_vxc(nxc_func_type* p, int np, double rho[], double sigma[], double * exc, double vrho[], double vsigma[]);
void nxc_mgga_exc_vxc(nxc_func_type* p, int np, double rho[],double sigma[], double lapl[],
    double tau[], double * exc, double vrho[], double vsigma[], double vlapl[],double vtau[]);
```
The arguments are defined in the same way as for Libxc with the notable exception that
sigma (and accordingly vsigma) can either be the reduced gradient or the gradient of the
density (and the corresponding potential term) depending on the parameter `gamma` in the `func_param` struct.

NeuralXC functionals require special treatment, as their dependency on localized atomic orbitals produces additional
terms when evaluating forces and stress. These corrections can be obtained with the method `nxc_lda_exc_vxc_fs`, which
should be called as the last step at the end of a converged SCF calcuation. When forces and stress aren't required
(e.g. during the SCF loop) it suffices to call `nxc_lda_exc_vxc` to evaluate the NeuralXC functional.

### Other methods
```c++
/**
* Check if GPU(cuda) is available
*/
int nxc_cuda_available();
void nxc_use_cuda(){
  defaults->useCuda();
}

const int LDA_TYPE=0;
const int GGA_TYPE=1;
const int MGGA_TYPE=2;
const int ATOMIC_TYPE=4;
int nxc_func_get_family(nxc_func_type* p);
int nxc_func_get_family_from_path(std::string model);
```
We provide two methods to check the type of a functional depending on whether the functional
has already been loaded and initialized (`nxc_func_get_family`) or whether we want to
check the functional type without loading it (`nxc_func_get_family_from_path`)


## Fortran Interface
The nomenclature used here strictly follows that of the C++ interface with the addition of
`_f90` in the function names. Function signatures reflect the fact that values cannot be returned to Fortran.
To Initialize functionals from fortran, two methods are availabl:
 - `nxc_f90_func_init_` to initialize grid based functionals (LDA, GGA, ...) for which
 only the model name/path has to be provided.
 - `nxc_f90_atmfunc_init_` to initialize NeuralXC functionals for which more information about the simulation box is required.

```c++
void nxc_f90_set_code_(int * code);
void nxc_f90_use_cuda_();
void nxc_f90_cuda_available(int * available);
int nxc_f90_atmfunc_init_(double  pos[], int * nua, double  cell[], int  grid[], int isa[],
            char symbols[], int * ns, char  modelpath[], int * pathlen, int myBox[], int* ierr);
int nxc_f90_func_init_(char  modelpath[], int * pathlen, int * ierr);
int nxc_f90_lda_exc_vxc_(int* np, double rho[], double exc [], double vrho[], int* ierr);
int nxc_f90_lda_exc_vxc_fs_(int* np, double rho[], double exc[], double vrho[],
                            double forces[], double stress[], int* ierr);
int nxc_f90_gga_exc_vxc_(int* np, double rho[], double sigma[], double exc [],
    double vrho[], double vsigma[], int* ierr);
int nxc_f90_mgga_exc_vxc_(int* np, double rho[], double sigma[], double lapl[], double tau[],
   double exc [], double vrho[], double vsigma[], double vlapl[], double vtau[], int* ierr);
void nxc_f90_func_get_family(int * family);
void nxc_f90_func_get_family_from_path_(char modelpath [], int * pathlen, int * family);
```

The trailing underscore in the function names is required for linking purposes and
has to be **dropped** when calling the function from fortran.

## pylibnxc

### Loading a model
```python
def LibNXCFunctional(name, **kwargs):
    """ Loads a Libnxc functional
    Parameters
    ----------
    name, string
        Either the name of a pre-defined functional or path to custom
        functional (will check path first and then resort to pre-defined functional)
    kind, string, optional {'grid', 'atomic'}
        default: 'grid', whether functional is grid kind (LDA, GGA etc.) or
        atomic (NeuralXC)

    Returns
    --------
    NXCFunctional
    """
```
### Initializing a model

If and only if the functional is of atomic (NeuralXC) kind it needs to
be initialized before usage. Keyword arguments passed to the initialize function are used to infer whether the calculation is done in periodic boundary conditions:

- Periodic boundary conditions with euclidean grid:
```python
def initialize(self, **kwargs):
    """Parameters
    ------------------
    unitcell, numpy.ndarray (3,3)
    	Unitcell in bohr
    grid, numpy.ndarray (3,)
    	Grid points per unitcell
    positions, numpy.ndarray (Natoms, 3)
    	atomic positions
    species, list string (Natoms)
    	atomic species (chem. symbols)
    """
```

- Non-periodic boundary conditions with custom grid:
```python
def initialize(self, **kwargs):
    """Parameters
    ------------------
    grid_coords, numpy.ndarray (Ngrid,3)
    	Grid point coordinates (in a.u.)
    grid_weights, numpy.ndarray (Nweights,)
    	Grid point weights for integration
    positions, numpy.ndarray (N, 3)
    	atomic positions
    species, list string (N)
    	atomic species (chem. symbols)
    """
```

### Evaluating a model

The model can be evaluated by calling `compute` on a `LibNXCFunctional` instance:

```python
def compute(self, inp, do_exc=True, do_vxc=True, **kwargs):
        """ Evaluate the functional on a given input

        Parameters
        ----------
        inp, dict of np.ndarrays
            Input electron density "rho" and its derivatives. Potential terms are
            calculated for all provided derivates. (valid keys: 'rho','sigma',
            'gamma','tau','lapl').
        do_exc, bool, optional
            no effect, only here for compatibility reasons
        do_vxc, bool, optional
            whether to compute the functional derivative(s) of the energy.
            default: True

        Returns:
        ---------
        output, dict
            Dictionary containing output values:
                - 'zk': energy per unit particle or total energy
                - 'vrho/vsigma/vtau' : potential terms
    """
```

If the functional type is atomic two additional keyword arguments can be provided:
  - `do_forces`: bool, Compute the pulay force corrections. The output dict will then
  contain an entry named `'forces'`.
  - `edens`: bool, Return energy per unit particle if `True`, total energy otherwise
In this case, instead of providing the electron density as `'rho'` the projected density
or ML-descriptors can be provided as `'c'` in which case the density projection step
is skipped but force corrections are not available. This might make sense for
codes for which analytical integrals over orbitals are available.



## Shipped functionals

The following functionals were introduced in

[1] *Nagai, Ryo, Ryosuke Akashi, and Osamu Sugino. "Completing density functional theory by machine learning hidden messages from molecules." npj Computational Materials 6.1 (2020): 1-8.*

Please consider citing the paper when using them.

- **LDA_HM**: NN-LSDA introduced in [1]
- **GGA_HM**: NN-GGA introduced in [1]
- **MGGA_HM**: NN-meta-GGA introduced in [1]


The following functionals are mainly included for testing purposes and should be used with care. For small molecules an accuracy of about 1 mHartree can be expected.

- **GGA_PBE**: Neural Network fitted to reproduce the famous PBE functional
- **GGA_X_PBE**: Exchange part of GGA_PBE
- **GGA_C_PBE**: Correlation part of GGA_PBE