# Lib**n**xc
=========================

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

``` pip install pyscf```

To unit test the C++/Fortran implementation [GoogleTest](https://github.com/google/googletest) is required.




## Quickstart

The functionality of Libnxc may be best explained by an example

```
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
0.1     -0.459795
0.2     -0.507272
0.3     -0.562437
0.4     -0.611048
0.5     -0.653416
```

The same program can be run from Fortran:

```
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

```
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