#ifndef NXC_H
#define NXC_H
#include <iostream>
#include <memory>

const int NXC_POLARIZED=2;
const int NXC_UNPOLARIZED=1;

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
  virtual void init(func_param fp, int nspin){};
  virtual void exc_vxc(int np, double rho[], double * exc, double vrho[]){};
  virtual void exc_vxc_fs(int np, double rho[], double * exc, double vrho[],
                        double forces[], double stress[]){};
protected:
  int nspin;
};

struct nxc_func_type{
  std::shared_ptr <NXCFunc> func;
};

std::shared_ptr<NXCFunc> get_functional(std::string model);
void nxc_func_init(nxc_func_type* p, std::string model, func_param fp, int nspin=NXC_UNPOLARIZED);
void nxc_func_init(nxc_func_type* p, std::string model, int nspin=NXC_UNPOLARIZED);

void nxc_lda_exc_vxc(nxc_func_type* p, int np, double rho[], double * exc, double vrho[]);
void nxc_lda_exc_vxc_fs(nxc_func_type* p, int np, double rho[], double * exc, double vrho[],
                        double forces[], double stress[]);

void nxc_gga_exc_vxc(nxc_func_type* p, int np, double rho[], double sigma[], double * exc, double vrho[]);
void nxc_mgga_exc_vxc(nxc_func_type* p, int np, double rho[],double sigma[], double lapl[], double tau[], double * exc, double vrho[]);

#endif
