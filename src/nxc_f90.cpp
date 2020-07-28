#include "nxc.h"
#include <iostream>
#include <memory>
#include "nxc_func.h"
extern "C"{

nxc_func_type nxc_func;

int nxc_f90_func_init_(double  pos[], int * nua, double  cell[], int  grid[], int isa[],
            char symbols[], int * ns, int * ierr, char  modelpath[], int * pathlen, int myBox[]) {

    std::string modeldir(modelpath);
    modeldir = modeldir.erase(*pathlen, std::string::npos);
    func_param fp = {pos, *nua, cell, grid, isa, symbols, *ns, myBox};
    nxc_func_init(&nxc_func, modeldir, fp);

  return 0;
}

// int nxc_f90_lda_exc_vxc_(double Vscf[], double rho[], int * np, double * Ex, int * ierr) {
int nxc_f90_lda_exc_vxc_(int* np, double rho[], double exc [], double vrho[], int* ierr) {

  nxc_lda_exc_vxc(&nxc_func, *np, rho, exc, vrho);
  return 0;
}

// int nxc_f90_lda_exc_vxc_fs_(double Vscf[], double rho[], int * np, double * Ex,
//                    double  forces[], double stress[], int * ierr) {
int nxc_f90_lda_exc_vxc_fs_(int* np, double rho[], double exc[], double vrho[],
                            double forces[], double stress[], int* ierr) {
  nxc_lda_exc_vxc_fs(&nxc_func, *np, rho, exc, vrho, forces, stress);
  return 0;

}

}
