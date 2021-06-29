#include "xc.h"
#include <iostream>
int main(int argc, char **argv){
  xc_func_type func;
  double rho[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
  double exc[5];
  double vrho[5];
  int func_id = 821;
  std::cout << "====== LIBNXC =====" << std::endl;
  xc_func_init(&func, func_id, XC_UNPOLARIZED);
  xc_lda_exc_vxc(&func, 5, rho, exc, vrho);
  for(int i=0; i<5;++i){
  	std::cout << rho[i] << "\t" << exc[i] << std::endl;
	exc[i] = 0;
  }
  xc_func_end(&func);

  std::cout << "====== LIBXC =====" << std::endl;
  xc_func_init(&func, 1, XC_UNPOLARIZED);
  xc_lda_exc_vxc(&func, 5, rho, exc, vrho);
  for(int i=0; i<5;++i){
  	std::cout << rho[i] << "\t" << exc[i] << std::endl;
  }
  xc_func_end(&func);
  return 0;
}
