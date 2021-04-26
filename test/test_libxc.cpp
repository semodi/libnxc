#include "xc.h"
#include <iostream>
int main(int argc, char **argv){
  xc_func_type func;
  double rho[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
  double sigma[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
  double exc[5];
  double vrho[5];
  double vsigma[5];
  int func_id = 801;
  std::cout << "====== UNPOLARIZED =====" << std::endl;
  xc_func_init(&func, func_id, XC_UNPOLARIZED);
  xc_gga_exc_vxc(&func, 5, rho, sigma, exc, vrho, vsigma);
  for(int i=0; i<5;++i){
  	std::cout << rho[i] << "\t" << exc[i] << std::endl;
  }
  xc_func_end(&func);

  double rho2[10] = {0.05, 0.1, 0.15, 0.2, 0.25, 0.05, 0.1, 0.15, 0.2, 0.25};
  double sigma2[15] = {0.2, 0.3, 0.4, 0.5, 0.6,0.2, 0.3, 0.4, 0.5, 0.6,0.2, 0.3, 0.4, 0.5, 0.6};
  for(int i=0; i<15;++i){
    sigma2[i] *= 0.25;
  }
  double exc2[5];
  double vrho2[10];
  double vsigma2[15];

  std::cout << "====== POLARIZED =====" << std::endl;
  xc_func_init(&func, func_id, XC_POLARIZED);
  xc_gga_exc_vxc(&func, 5, rho2, sigma2, exc2, vrho2, vsigma2);
  for(int i=0; i<5;++i){
  	std::cout << rho[i] << "\t" << exc2[i] << std::endl;
  }
  xc_func_end(&func);
  return 0;
}
