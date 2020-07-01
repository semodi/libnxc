#include <torch/script.h> // One-stop header.
#include "nxc.h"
#include "nxc_func.h"
#include <iostream>
#include <memory>


void nxc_func_init(nxc_func_type* p, std::string modeldir, func_param fp){
    std::shared_ptr<NXCFunc> f = get_functional(modeldir);
    f->init(fp);
    p->func = f;
}

void nxc_lda_exc_vxc(nxc_func_type* p, int np, double rho[], double * exc, double vrho[]){
  p->func->exc_vxc(np, rho, exc, vrho);
}

void nxc_lda_exc_vxc_fs(nxc_func_type* p, int np, double rho[], double * exc, double vrho[],
                        double forces[], double stress[]){

    p->func->exc_vxc_fs(np, rho, exc, vrho, forces, stress) ;
  }
