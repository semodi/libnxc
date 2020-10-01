#include <iostream>
#include "nxc.h"

int load_model_nopar(){
    nxc_func_type p;
    nxc_func_init(&p, "../test.jit");
    nxc_func_init(&p, "../test_agn.jit");
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
    nxc_func_init(&p, "../test_agn.jit", fp);
    return 0;
}
