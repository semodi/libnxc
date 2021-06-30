#ifdef LIBXC
#include "libxc.h"
void test_func(int func_id, double exc[], double vrho[],
               double vsigma[], double vtau[]){

  xc_func_type func;
  double rho[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
  double sigma[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
  double lapl[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  double tau[5] = {0.3, 0.3, 0.4, 0.5, 0.6};
  double vlapl[5];
  if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
    fprintf(stderr, "Functional '%d' not found\n", func_id);
  }

  switch(func.info->family)
    {
    case XC_FAMILY_LDA:
      xc_lda_exc_vxc(&func, 5, rho, exc, vrho);
      break;
    case XC_FAMILY_GGA:
    case XC_FAMILY_HYB_GGA:
      xc_gga_exc_vxc(&func, 5, rho, sigma, exc, vrho, vsigma);
      break;
    case XC_FAMILY_MGGA:
      xc_mgga_exc_vxc(&func, 5, rho, sigma,lapl,tau, exc, vrho, vsigma,vlapl,vtau);
      break;
    }

  printf(func.info->name);
  printf("\n");

  xc_func_end(&func);

}
#endif
