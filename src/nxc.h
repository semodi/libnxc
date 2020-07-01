#ifndef NXC_H
#define NXC_H
#include <iostream>
#include <torch/script.h> // One-stop header.
#include "nxc_func.h"



void nxc_func_init(nxc_func_type* p, std::string modeldir, func_param fp);
void nxc_lda_exc_vxc(nxc_func_type* p, int np, double rho[], double * exc, double vrho[]);
void nxc_lda_exc_vxc_fs(nxc_func_type* p, int np, double rho[], double * exc, double vrho[],
                        double forces[], double stress[]);

#endif
