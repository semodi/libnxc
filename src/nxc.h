#ifndef NXC_H
#define NXC_H
#include <iostream>
#include <memory>
//#include <torch/script.h> // One-stop header.
//#include "nxc_func.h"


struct func_param{
  double * pos; // atomic positions
  int nua; // number of atoms (pos.size())
  double * cell; // lattice vectors
  int * grid; // number of grid points for each LV
  int * isa; // species index for every atom  (relates to symbols array)
  char * symbols; //distinct symbols
  int ns; // symbols.size()
  int * myBox; // box in simulation cell (used mainly for MPI decomposition)
  int edens;
};

class NXCFunc {

public:
  NXCFunc(){};
  virtual void init(){};
  virtual void init(func_param fp){};
  virtual void exc_vxc(int np, double rho[], double * exc, double vrho[])=0;
  virtual void exc_vxc_fs(int np, double rho[], double * exc, double vrho[],
                        double forces[], double stress[])=0;
};

struct nxc_func_type{
  std::shared_ptr <NXCFunc> func;
};

void nxc_func_init(nxc_func_type* p, std::string modeldir, func_param fp);
void nxc_func_init(nxc_func_type* p, std::string modeldir);
void nxc_lda_exc_vxc(nxc_func_type* p, int np, double rho[], double * exc, double vrho[]);
void nxc_lda_exc_vxc_fs(nxc_func_type* p, int np, double rho[], double * exc, double vrho[],
                        double forces[], double stress[]);


#endif
