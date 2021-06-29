#include <iostream>
#include "nxc.h"

int main()
{
  nxc_func_type p;
  func_param fp;
  int nspin = NXC_UNPOLARIZED;

  double rho[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
  double sigma[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
  double exc[5];

  double vrho[5];

  nxc_func_init(&p, "LDA_X_NUEG", fp, nspin);
//  nxc_func_init(&p, "LDA_HM", fp, nspin);

  nxc_lda_exc_vxc(&p, 5, rho, exc, vrho);


  for(int i=0; i<5;++i){
  	std::cout << rho[i] << "\t" << exc[i] << std::endl;
  }

  return 0;
}
