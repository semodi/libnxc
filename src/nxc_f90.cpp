#include "nxc_f90.h"
#include "nxc.h"
#include <iostream>
#include <memory>
#include "nxc_func.h"

extern "C"{

nxc_func_type nxc_func;
bool func_set=false;
void nxc_f90_set_code_(int * code){
  nxc_set_code(*code);
}
void nxc_f90_use_cuda_(){
  nxc_use_cuda();
}
void nxc_f90_cuda_available(int * available){
  *available = nxc_cuda_available();
}
void nxc_f90_atmfunc_init_(double  pos[], int * nua, double  cell[], int  grid[], int isa[],
            char symbols[], int * ns, char  modelpath[], int * pathlen, int myBox[], int* ierr){

    std::string modeldir(modelpath);
    modeldir = modeldir.erase(*pathlen, std::string::npos);
    func_param fp = {pos, *nua, cell, grid, isa, symbols, *ns, myBox};
    nxc_func_init(&nxc_func, modeldir, fp);
    func_set=true;
}

void nxc_f90_func_init_(char  modelpath[], int * pathlen, int * ierr){
    std::string modeldir(modelpath);
    modeldir = modeldir.erase(*pathlen, std::string::npos);
    func_param fp;
    nxc_func_init(&nxc_func, modeldir, fp);
    func_set=true;
}
void nxc_f90_lda_exc_vxc_(int* np, double rho[], double exc [], double vrho[], int* ierr) {
  nxc_lda_exc_vxc(&nxc_func, *np, rho, exc, vrho);
}

void nxc_f90_lda_exc_vxc_fs_(int* np, double rho[], double exc[], double vrho[],
                            double forces[], double stress[], int* ierr) {
  nxc_lda_exc_vxc_fs(&nxc_func, *np, rho, exc, vrho, forces, stress);

}

void nxc_f90_gga_exc_vxc_(int* np, double rho[], double sigma[], double exc [],
    double vrho[], double vsigma[], int* ierr) {
  nxc_gga_exc_vxc(&nxc_func, *np, rho, sigma, exc, vrho, vsigma);
}

void nxc_f90_mgga_exc_vxc_(int* np, double rho[], double sigma[], double lapl[], double tau[],
   double exc [], double vrho[], double vsigma[], double vlapl[], double vtau[], int* ierr) {
  nxc_mgga_exc_vxc(&nxc_func, *np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau);
}

void nxc_f90_func_get_family_(int * family){
  if (func_set){
    *family = nxc_func_get_family(&nxc_func);
  }else{
    *family = -1;
  }
}
void nxc_f90_func_get_family_from_path_(char modelpath [], int * pathlen, int * family){
  std::string modeldir(modelpath);
  modeldir = modeldir.erase(*pathlen, std::string::npos);
  *family = nxc_func_get_family_from_path(modeldir);
}
}
