#ifndef NXCF_H
#define NXCF_H
extern "C"{

void nxc_f90_set_code_(int * code);
void nxc_f90_use_cuda_();
void nxc_f90_cuda_available(int * available);


void nxc_f90_atmfunc_init_(double  pos[], int * nua, double  cell[], int  grid[], int isa[],
            char symbols[], int * ns, char  modelpath[], int * pathlen, int myBox[], int* ierr);

void nxc_f90_func_init_(char  modelpath[], int * pathlen, int * ierr);

/**
* Evaluates the functional on provided density if functional is LDA type. This includes atomic functionals
* that only depend on the local density.
*
* @param[in] np number of grid points (size of rho)
* @param[in] rho electron density
* @param[(in), out] exc energy density. If fp.edens = 0, exc[0] contains energy.
* @param[(in), out] vrho dE/drho
* @param[out] ierr
*/
void nxc_f90_lda_exc_vxc_(int* np, double rho[], double exc [], double vrho[], int* ierr);

/**
* Evaluates the functional on provided density if functional is LDA type. This includes atomic functionals
* that only depend on the local density. Corrects forces and stress.
*
* @param[in] np number of grid points (size of rho)
* @param[in] rho electron density
* @param[(in), out] exc energy density. If fp.edens = 0, exc[0] contains energy.
* @param[(in), out] vrho dE/drho
* @param[in, out] forces
* @param[in, out] stress
* @param[out] ierr
*/
void nxc_f90_lda_exc_vxc_fs_(int* np, double rho[], double exc[], double vrho[],
                            double forces[], double stress[], int* ierr);

/**
* Evaluates the functional on provided density if functional is GGA type.
*
* @param[in] np number of grid points (size of rho)
* @param[in] rho electron density
* @param[in] sigma gradient of electron density
* @param[(in), out] exc energy density. If fp.edens = 0, exc[0] contains energy.
* @param[(in), out] vrho dE/drho
* @param[(in), out] vsigma dE/dsigma
* @param[out] ierr
*/
void nxc_f90_gga_exc_vxc_(int* np, double rho[], double sigma[], double exc [],
    double vrho[], double vsigma[], int* ierr);


/**
* Evaluates the functional on provided density if functional is MGGA type.
*
* @param[in] np number of grid points (size of rho)
* @param[in] rho electron density
* @param[in] sigma gradient of electron density
* @param[in] lapl laplacian of electron density
* @param[in] tau kinetic energy density
* @param[(in), out] exc energy density. If fp.edens = 0, exc[0] contains energy.
* @param[(in), out] vrho dE/drho
* @param[(in), out] vsigma dE/dsigma
* @param[(in), out] vlapl dE/dlapl
* @param[(in), out] vtau dE/dtau
* @param[out] ierr
*/
void nxc_f90_mgga_exc_vxc_(int* np, double rho[], double sigma[], double lapl[], double tau[],
   double exc [], double vrho[], double vsigma[], double vlapl[], double vtau[], int* ierr);

void nxc_f90_func_get_family(int * family);
void nxc_f90_func_get_family_from_path_(char modelpath [], int * pathlen, int * family);
}
#endif
