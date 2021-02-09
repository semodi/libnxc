#include <iostream>
#include <memory>
#include "nxc_c.h"
#include "nxc.h"
#include "funcs.h"
// #include "nxc_func.h"
nxc_func_type nxc_func_c;

void nxc_c_gga_exc_vxc(int * xc_func, int np, double rho[], double sigma[],
  double exc [], GGA_OUT_PARAMS_NOEXC(double *)){
  nxc_gga_exc_vxc(&nxc_func_c, np, rho, sigma, exc, vrho, vsigma);
}

void nxc_c_mgga_exc_vxc(int * xc_func, int np, double rho[], double sigma[], double lapl[],
  double tau[], double exc [], MGGA_OUT_PARAMS_NOEXC(double *)){
  nxc_mgga_exc_vxc(&nxc_func_c, np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl_rho, vtau);
}

void nxc_c_xc_init(int xc_func){
  func_param fp;
  nxc_func_init(&nxc_func_c, funcs.at(xc_func), fp, NXC_UNPOLARIZED);
}
