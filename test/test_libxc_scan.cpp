#include "xc.h"
#include <iostream>
int main(int argc, char **argv){
  xc_func_type func;
  double rho[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
  double sigma[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
// double sigma[5] = {2, 3, 4, 5, 6};
  double lapl[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
  double tau[5] = {0.3, 0.3, 0.4, 0.5, 0.6};
  double exc[5];
  double vrho[5];
  double vsigma[5];
  double vlapl[5];
  double vtau[5];
  int func_id = 815;
  std::cout << "====== LIBNXC =====" << std::endl;
  xc_func_init(&func, func_id, XC_UNPOLARIZED);
  xc_mgga_exc_vxc(&func, 5, rho, sigma, lapl, tau, exc, vrho, vsigma,vlapl, vtau);
  for(int i=0; i<5;++i){
  	std::cout << rho[i] << "\t" << exc[i] << std::endl;
	exc[i] = 0;
  }
  xc_func_end(&func);

  std::cout << "====== LIBXC =====" << std::endl;
  xc_func_init(&func, 263, XC_UNPOLARIZED);
  xc_mgga_exc_vxc(&func, 5, rho, sigma, lapl, tau, exc, vrho, vsigma,vlapl, vtau);
  for(int i=0; i<5;++i){
  	std::cout << rho[i] << "\t" << exc[i] << std::endl;
  }
  xc_func_end(&func);
  return 0;
}
