#include "atomic.h"

int load_model_nopar(){
    nxc_func_type p;
    nxc_func_init(&p, "../test.jit");
    return 0;
}

int load_model(){
    nxc_func_type p;

    double pos[3] = {0.1,0.0, 0.2};
    int nua = 1;
    double cell[9] = {10, 0,0,0,10,0,0,0,10};
    int grid[3] = {100,100,100};
    char symbols[1] = {'O'};
    int isa[1] = {0};
    int ns = 1;
    int myBox[6] = {0, 10,0,10,0,10};
    int edens = 1;

    func_param fp = {pos, nua, cell, grid, isa, symbols, ns, myBox, edens};
    nxc_func_init(&p, "../test.jit", fp);
    return 0;
}

double run_model(int *myBox, bool cuda){
    nxc_func_type p;

    double pos[3] = {0., 0., 0.};
    int nua = 1;
    double cell[9] = {10, 0,0,0,10,0,0,0,10};
    int grid[3] = {10,10,10};
    char symbols[2] = {'O',' '};
    int isa[1] = {1};
    int ns = 1;
    int edens = 0;

    nxc_set_code(SIESTA_CODE);
    if (cuda){
      // func_param fp = {pos, nua, cell, grid, isa, symbols, ns, myBox, edens, 1, 1};
      func_param fp = {pos, nua, cell, grid, isa, symbols, ns, myBox};
      fp.cuda=1;
      nxc_func_init(&p, "../test.cuda.jit", fp);
    }else{
      func_param fp = {pos, nua, cell, grid, isa, symbols, ns, myBox};
      nxc_func_init(&p, "../test.jit", fp);
    }

    int np = 1;
    for (int i =0; i<3; i++){
      np = np * (myBox[2*i+1] - myBox[2*i] + 1);
    }
    double rho[1000] = {0.1};
    double Exc[1] = {0};
    double vrho[1000];
    nxc_lda_exc_vxc(&p, np, rho, Exc, vrho);
    return Exc[0];
}
