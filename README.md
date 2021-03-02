[![Documentation Status](https://readthedocs.org/projects/libnxc/badge/?version=latest)](https://libnxc.readthedocs.io/en/latest/?badge=latest)
![Python Unit Testing](https://github.com/semodi/libnxc/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/semodi/libnxc/branch/master/graph/badge.svg?token=PM061YXF17)](https://codecov.io/gh/semodi/libnxc)
# Lib**n**xc

Libnxc is a libary to use **machine learned** exchange-correlation functionals for density functional theory.
All common functional types (LDA, GGA, metaGGA) as well as NeuralXC type functionals  are supported.
Libnxc is written in C++ and has Fortran bindings. An implementation in Python, `pylibnxc` is also available.
Libnxc is inspired by Libxc, mirroring as closely  as possible its API. In doing so, the integration of Libnxc in electronic structure codes that use Libxc should be straightforward.
Libnxc can utilize multi-processing through MPI and model inference on GPUs through CUDA is supported as well.
Although the primary motivation for Libnxc was to add support for neural network based functionals, other types of models can be used as well.
As long as the following requirements are fulfilled, models can be used by Libnxc:

  1. The model has to be implemented in PyTorch and serialized into a TorchScript model (e.g. with ``torch.jit.trace``)
  2. The model input and output has to follow the form specified in the [Documentation](https://libnxc.readthedocs.io/en/latest/functionals.html)

The serialized model is regarded as a containerized black box by Libnxc.
Thus, even simple polynomial models can be implemented and evaluated.
While not replacing hard-coded functionals such as the ones employed by Libxc and directly by DFT codes,
this approach provides several advantages:

  - **Fast experimentation**: Functionals can be quickly implemented and used in a 'plug-and-play' manner
  - **Automatic differentiation**: PyTorch takes care of calculating all derivative terms needed in the exchange-correlation potential.
  - **Native GPU support**: PyTorch is designed to be run on GPUs using CUDA. This extends to serialized TorchScript models, therefore
    evaluating libnxc functionals on GPUs is straightforward.


Table of Contents
=================

   * [Lib<strong>n</strong>xc](#libnxc)
      * [Dependencies](#dependencies)
      * [Installation](#Installation)
      * [Quickstart](#quickstart)
      * [Using Pylibnxc with PySCF](#using-pylibnxc-with-pyscf)
      * [Using Libnxc with Libxc](#using-libnxc-with-libxc)
      * [Shipped functionals](#shipped-functionals)

## Dependencies

Libnxc depends on PyTorch. The C++/Fortran implementation requires that libtorch is made available at compile time, pylibnxc depends on pytorch.
Both libraries are available for download on the [pytorch website](https://pytorch.org/get-started/locally/).

pylibnxc further depends on numpy, which is a requirement for pytorch and should therefore be installed automatically.

**Optional dependencies:**

For unit testing, pylibnxc currently requires [pyscf](https://sunqm.github.io/pyscf/install.html) which can be obtained through

`pip install pyscf`

To unit test the C++/Fortran implementation [GoogleTest](https://github.com/google/googletest) is required.

## Installation

To compile Libnxc:

1. Create a work directory and copy `utils/Makefile` into it (this can be done quickly with `sh config.sh`)
1. Adjust arch.make according to your system
1. Change into the work directory and run `make`. The default target is `make all` which will build the library `libnxc.so`
  and if `LIBXCDIR` is set in `arch.make` will create the necessary symbolic links in your Libxc installation

To install pylibnxc:

pylibnxc operates independent of Libnxc and can be installed from within the root directory with

`pip install -e .`


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
0.1     -0.459810
0.2     -0.507294
0.3     -0.562470
0.4     -0.611119
0.5     -0.653536
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

## Using Pylibnxc with PySCF

Using Pylibnxc in PySCF is as simple as changing two lines of code:

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
PBE (see [Shipped functionals](#shipped-functionals).

For unrestricted Kohn-Sham calculations `pylibnxc.pyscf.UKS` is available as well.
The `nxc` keyword supports mixing of functionals similar to pyscf, e.g.
`nxc ='0.25*HF + 0.75*GGA_X_PBE, GGA_C_PBE'` would correspond to a neural network version
of PBE0.
Currently, mixing of libxc functionals with libnxc functionals is not supported.

## Using Libnxc with Libxc

The simplest way to use machine learned functionals in existing electronic
structure codes is by linking Libxc with Libnxc. This way any code that supports
Libxc functionals (Quantum Espresso, CP2K, Vasp, Psi4, ...) has access to Libnxc routines.
This comes with two caveats:
1) Only grid based functionals are supported through this solution. There is currently
no way to pass the additional information needed by Libnxc (atomic positions, unit cell size etc.)
through Libxc.
2) Without adjusting the electronic structure code, using Libnxc with Neural Network functionals
may be significantly slower than calling traditional functionals. This is because some codes pass
the density grid point by grid point. This is fine for traditional functionals, but to efficiently
evaluate NN functionals, grid points should be processed in a vectorized fashion. Codes like
Quantum espresso already do so, but others, like CP2K, will need to be modified to ensure efficient
evaluation.

Using Libnxc from within Libxc is straightforward:

- Set the `LIBXCDIR` variable in arch.make to the
location of the libxc `src` directory.
- In order to be able to run unit tests using Libxc `LIBXC_INCLUDE`
and `LIBXC_LD` need to be set accordingly.
- From within the build directory `make libxc` will create symbolic links of the required
Libnxc source files within the Libxc `src` directory and copy a modified `Makefile.am` that includes
instructions to link to Libnxc. In some cases automake needs to be run in the Libxc directory.
- After successfully compiling Libnxc, Libxc should be re-compiled to include the new functionals.

## Shipped functionals

The following functionals were introduced in

[1] *Nagai, Ryo, Ryosuke Akashi, and Osamu Sugino. "Completing density functional theory by machine learning hidden messages from molecules." npj Computational Materials 6.1 (2020): 1-8.*

Please consider citing the paper when using them.

- **LDA_HM**: NN-LSDA introduced in [1]
- **GGA_HM**: NN-GGA introduced in [1]
- **MGGA_HM**: NN-meta-GGA introduced in [1]


The following functionals are mainly included for testing purposes and should be used with care. For small molecules an accuracy of about 1 mHartree can be expected.

- **GGA_PBE**: Neural Network fitted to reproduce the popular PBE functional
- **GGA_X_PBE**: Exchange part of GGA_PBE
- **GGA_C_PBE**: Correlation part of GGA_PBE
