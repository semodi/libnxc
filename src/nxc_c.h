#ifndef NXCC_H
#define NXCC_H
#ifdef __cplusplus
extern "C"{
#endif

#define LDA_OUT_PARAMS_NOEXC(P_)                                \
  P_ vrho,                                                  \
  P_ v2rho2,          \
  P_ v3rho3,         \
  P_ v4rho4

// Adapted from libxc
#define GGA_OUT_PARAMS_NOEXC(P_)                                          \
  P_ vrho, P_ vsigma,                                                      \
  P_ v2rho2, P_ v2rhosigma, P_ v2sigma2,                                   \
  P_ v3rho3, P_ v3rho2sigma, P_ v3rhosigma2, P_ v3sigma3,                  \
  P_ v4rho4, P_ v4rho3sigma, P_ v4rho2sigma2, P_ v4rhosigma3, P_ v4sigma4  \

// Adapted from libxc
#define MGGA_OUT_PARAMS_NOEXC(P_)                                         \
        P_ vrho, P_ vsigma, P_ vlapl_rho, P_ vtau,                      \
	       P_ v2rho2, P_ v2sigma2, P_ v2lapl2, P_ v2tau2,                  \
	       P_ v2rhosigma, P_ v2rholapl, P_ v2rhotau,                     \
	       P_ v2sigmalapl, P_ v2sigmatau, P_ v2lapltau                 \

void nxc_c_lda_exc_vxc(int * xc_func, int np, double rho[], double exc [], LDA_OUT_PARAMS_NOEXC(double *));
void nxc_c_gga_exc_vxc(int * xc_func, int np, double rho[], double sigma[], double exc [], GGA_OUT_PARAMS_NOEXC(double *));
void nxc_c_mgga_exc_vxc(int * xc_func, int np, double rho[], double sigma[],
  double lapl[], double tau[], double exc [], MGGA_OUT_PARAMS_NOEXC(double *));

int nxc_c_xc_init(int xc_func, int nspin);
#ifdef __cplusplus
}
#endif
#endif
