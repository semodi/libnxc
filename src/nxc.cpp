#include <torch/script.h> // One-stop header.
#include "nxc.h"
#include "nxc_func.h"
#include <iostream>
#include <memory>


void nxc_func_init(nxc_func_type* p, std::string model, func_param fp, int nspin){
    std::shared_ptr<NXCFunc> f = get_functional(model);
    f->init(fp, nspin);
    p->func = f;
}

void nxc_func_init(nxc_func_type* p, std::string model, int nspin){
    std::shared_ptr<NXCFunc> f = get_functional(model);
    p->func = f;
}

void nxc_lda_exc_vxc(nxc_func_type* p, int np, double rho[], double exc[], double vrho[]){
  p->func->exc_vxc(np, rho, exc, vrho);
}

void nxc_lda_exc_vxc_fs(nxc_func_type* p, int np, double rho[], double exc[], double vrho[],
                        double forces[], double stress[]){

    p->func->exc_vxc_fs(np, rho, exc, vrho, forces, stress) ;
  }
