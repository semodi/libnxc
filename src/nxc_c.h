#ifndef NXCC_H
#define NXCC_H
#ifdef __cplusplus
extern "C"{
#endif

#define XC_GGA_X_NXC          801 /* Machine learned GGA X functional                */
#define XC_GGA_C_NXC          802 /* Machine learned GGA C functional                */
#define XC_GGA_XC_NXC         803 /* Machine learned GGA XC functional               */

#define GGA_OUT_PARAMS_NOEXC(P_)                                          \
  P_ vrho, P_ vsigma,                                                      \
  P_ v2rho2, P_ v2rhosigma, P_ v2sigma2,                                   \
  P_ v3rho3, P_ v3rho2sigma, P_ v3rhosigma2, P_ v3sigma3,                  \
  P_ v4rho4, P_ v4rho3sigma, P_ v4rho2sigma2, P_ v4rhosigma3, P_ v4sigma4  \

void nxc_c_gga_exc_vxc(int * xc_func, int np, double rho[], double sigma[], double exc [], GGA_OUT_PARAMS_NOEXC(double *));
void nxc_c_gga_xc_init(int xc_func);
#ifdef __cplusplus
}
#endif
#endif
