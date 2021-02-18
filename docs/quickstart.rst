Quickstart
=======================
The functionality of Libnxc may be best explained by an example::

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

- ``nxc_func_type p`` stores the functional
- ``func_param fp`` stores parameters that determine how the functional is evaluated and how Libnxc communicates with the external code. For now, it suffices to know that these parameters are set to sensible default values. We will return to it later.
- ``nxc_func_init(&p, "GGA_PBE", fp, nspin)`` loads the functional "GGA_PBE", the second argument can either be the name of a shipped functional (see below for a list thereof) or a path to a serialized TorchScript model (at the moment custom models are only available in pylibnxc).
- ``nxc_gga_exc_vxc(&p, 5, rho, sigma, exc, vrho, vsigma)`` evaluates the functional and stores the energy per unit particle, and the potential terms $\delta E/ \delta \rho(\mathbf r)$ and $\delta E/ \delta \sigma(\mathbf r)$ in `exc`, `vrho` and `vsigma`, respectively.



Running the program above should produce the following output::

  0.1     -0.459810
  0.2     -0.507294
  0.3     -0.562470
  0.4     -0.611119
  0.5     -0.653536


The same program can be run from Fortran:::


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


... and Python::


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
