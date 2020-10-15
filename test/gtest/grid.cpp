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
    // double vrho_up[100];
    // double exc_up[100];
    nxc_lda_exc_vxc(&p, np, rho_up, exc_up, vrho_up);

    nspin = NXC_POLARIZED;
    nxc_func_init(&p,"HM_LDA", fp, nspin);
    double rho_p[200];
    for( int is=0; is<2; ++is){
      for( int i =0; i< np; ++i){
        rho_p[is*np+i] = double(i+10  )/200.0;
      }
    }
    // double vrho_p[200];
    // double exc_p[100];
    nxc_lda_exc_vxc(&p, np, rho_p, exc_p, vrho_p);
}
