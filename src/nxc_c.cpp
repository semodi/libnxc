#include "nxc_c.h"
#include "nxc.h"
#include <iostream>
#include <memory>
#include "nxc_func.h"


#ifdef __cplusplus
extern "C"{
#endif

nxc_func_type nxc_func_c;
bool func_set_c=false;
void nxc_c_gga_exc_vxc(int * xc_func, int np, double rho[], double sigma[],
  double exc [], GGA_OUT_PARAMS_NOEXC(double *)){
  nxc_gga_exc_vxc(&nxc_func_c, np, rho, sigma, exc, vrho, vsigma);
    }

void nxc_c_gga_x_init(int* xc_func){
  func_param fp;
  nxc_func_init(&nxc_func_c, "GGA_X_PBE", fp, NXC_UNPOLARIZED);
}
void nxc_c_gga_c_init(int* xc_func){
  func_param fp;
  nxc_func_init(&nxc_func_c, "GGA_C_PBE", fp, NXC_UNPOLARIZED);
}
void nxc_c_gga_xc_init(int* xc_func){
  func_param fp;
  nxc_func_init(&nxc_func_c, "GGA_PBE", fp, NXC_UNPOLARIZED);
}

#ifdef __cplusplus
}
#endif
