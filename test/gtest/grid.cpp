#include "grid.h"

void load_hm_models(){
    nxc_func_type p;
    func_param fp;
    nxc_func_init(&p,"HM_LDA", fp);
    nxc_func_init(&p,"HM_GGA", fp);
    nxc_func_init(&p,"HM_MGGA", fp);
}

void test_hm_lda(double vrho_up[],double exc_up[], double vrho_p[], double exc_p[]){
    nxc_func_type p;
    func_param fp;
    int nspin = NXC_UNPOLARIZED;
    nxc_func_init(&p,"HM_LDA", fp, nspin);
    const int np = 100;
    double rho_up[100];
    for( int i =0; i< np; ++i){
      rho_up[i] = double(i+10)/100.0;
    }
    nxc_lda_exc_vxc(&p, np, rho_up, exc_up, vrho_up);

    nspin = NXC_POLARIZED;
    nxc_func_init(&p,"HM_LDA", fp, nspin);
    double rho_p[200];
    for( int is=0; is<2; ++is){
      for( int i =0; i< np; ++i){
        rho_p[is*np+i] = double(i+10)/200.0;
      }
    }
    nxc_lda_exc_vxc(&p, np, rho_p, exc_p, vrho_p);
}

void test_hm_gga(double vrho_up[], double vsigma_up[], double exc_up[],
                 double vrho_p[], double vsigma_p[], double exc_p[]){
    nxc_func_type p;
    func_param fp;
    int nspin = NXC_UNPOLARIZED;
    nxc_func_init(&p,"HM_GGA", fp, nspin);
    const int np = 100;
    double rho_up[100];
    double sigma_up[100];
    for( int i =0; i< np; ++i){
      rho_up[i] = double(i+10)/100.0;
      sigma_up[i] = double(i+10)/100.0;
    }
    nxc_gga_exc_vxc(&p, np, rho_up, sigma_up, exc_up, vrho_up, vsigma_up);

    nspin = NXC_POLARIZED;
    nxc_func_init(&p,"HM_GGA", fp, nspin);
    double rho_p[200];
    double sigma_p[300];
    for( int is=0; is<2; ++is){
      for( int i =0; i< np; ++i){
        rho_p[is*np+i] = double(i+10)/200.0;
      }
    }
    for( int is=0; is<3; ++is){
      for( int i =0; i< np; ++i){
        sigma_p[is*np+i] = sigma_up[i]*0.25;
      }
    }
    nxc_gga_exc_vxc(&p, np, rho_p, sigma_p, exc_p, vrho_p, vsigma_p);
}
