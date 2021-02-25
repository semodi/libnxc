#include <iostream>
#include "xc.h"
int main(){
  std::cout << "Hello World" << std::endl;
  std::cout << "Hello World" << std::endl;
  std::cout << "Hello World" << std::endl;
  xc_func_type func;
  std::cout << "Hello World" << std::endl;
  double rho[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
  double sigma[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
  double lapl[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
  double tau[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
  double vlapl[5];
  double exc[5];
  double vrho[5];
  double vsigma[5];
  double vtau[5];
  int func_id = 101;
  std::cout << func.info->name << std::endl;
  std::cout << func.info->family << std::endl;
  switch(func.info->family)
    {
    case XC_FAMILY_LDA:
      xc_lda_exc(&func, 5, rho, exc);
      break;
    case XC_FAMILY_GGA:
    case XC_FAMILY_HYB_GGA:
      xc_gga_exc_vxc(&func, 5, rho, sigma, exc, vrho, vsigma);
      break;
    case XC_FAMILY_MGGA:
      xc_mgga_exc_vxc(&func, 5, rho, sigma,lapl,tau, exc,
          vrho, vsigma,vlapl,vtau);
      break;
    }


  xc_func_end(&func);

}
