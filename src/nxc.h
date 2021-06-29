#ifndef NXC_H
#define NXC_H
#include <iostream>
#include <memory>
#include "defaults.h"
const int NXC_POLARIZED=2;
const int NXC_UNPOLARIZED=1;

/**
* Parameters to be passed to functional
*
* @param pos atomic positions
* @param nua number of atoms
* @param cell lattice vectors
* @param grid number of grid points for each lattice vector
* @param isa species index for every atom
* @param symbols ditinct symbols
* @param ns symbols.size()
* @param myBox box in simulation cell (used mainly for MPI decomposition)
* @param edens 0: return total energy, 1: return energy density  (default: 1)
* @param add 0: set return values 1: add return values (default: 1)
* @param cuda 0: use cpu 1: use gpu (default: 0)
*/
struct func_param{
  double * pos; // atomic positions
  int nua; // number of atoms (pos.size())
  double * cell; // lattice vectors
  int * grid; // number of grid points for each LV
  int * isa; // species index for every atom  (relates to symbols array)
  char * symbols; //distinct symbols
  int ns; // symbols.size()
  int * myBox; // box in simulation cell (used mainly for MPI decomposition)
  int edens = defaults->edens;
  int add = defaults->add;
  int cuda = defaults->cuda;
  int gamma = defaults->gamma;
};


/**
* Base class from which all other functionals (both grid based and atomic) are derived
*/
class NXCFunc {

public:
  NXCFunc(){};
  virtual void init(){};
  virtual void init(func_param fp, int nspin){};
  virtual void exc_vxc(int np, double rho[], double * exc, double vrho[]){};
  virtual void exc_vxc(int np, double rho[], double sigma[], double * exc, double vrho[], double vsigma[]){};
  virtual void exc_vxc(int np, double rho[], double sigma[], double lapl[], double tau[],
     double * exc, double vrho[], double vsigma[], double vlapl[], double vtau[]){};
  virtual void exc_vxc_fs(int np, double rho[], double * exc, double vrho[],
                        double forces[], double stress[]){};
  virtual void to_cuda(){};
  virtual int get_family(){return -1;};
protected:
  int nspin;
  int device;
};

struct nxc_func_type{
  std::shared_ptr <NXCFunc> func;
};


std::shared_ptr<NXCFunc> get_functional(std::string model);

void nxc_set_code(int code);
void nxc_use_cuda();
/**
* Initializes functional
*
* @param[out] p loaded functional
* @param[in] model string containing either model path or name
* @param[in, optional] fp functional parameters
* @param[in, optional] nspin spin polarized/unpolarized calcuation (default NXC_UNPOLARIZED)
*/
int nxc_func_init(nxc_func_type* p, std::string model, func_param fp, int nspin=NXC_UNPOLARIZED);
int nxc_func_init(nxc_func_type* p, std::string model, int nspin=NXC_UNPOLARIZED);

/**
* Evaluates the functional on provided density if functional is LDA type. This includes atomic functionals
* that only depend on the local density.
*
* @param[in] p functional to evaluate
* @param[in] np number of grid points (size of rho)
* @param[in] rho electron density
* @param[(in), out] exc energy density. If fp.edens = 0, exc[0] contains energy.
* @param[(in), out] vrho dE/drho
*/
void nxc_lda_exc_vxc(nxc_func_type* p, int np, double rho[], double * exc, double vrho[]);

/**
* Evaluates the functional on provided density if functional is LDA type. This includes atomic functionals
* that only depend on the local density. Corrects forces and stress.
*
* @param[in] p functional to evaluate
* @param[in] np number of grid points (size of rho)
* @param[in] rho electron density
* @param[(in), out] exc energy density. If fp.edens = 0, exc[0] contains energy.
* @param[(in), out] vrho dE/drho
* @param[in, out] forces
* @param[in, out] stress
*/
void nxc_lda_exc_vxc_fs(nxc_func_type* p, int np, double rho[], double * exc, double vrho[],
                        double forces[], double stress[]);

/**
* Evaluates the functional on provided density if functional is GGA type.
*
* @param[in] p functional to evaluate
* @param[in] np number of grid points (size of rho)
* @param[in] rho electron density
* @param[in] sigma gradient of electron density
* @param[(in), out] exc energy density. If fp.edens = 0, exc[0] contains energy.
* @param[(in), out] vrho dE/drho
* @param[(in), out] vsigma dE/dsigma
*/
void nxc_gga_exc_vxc(nxc_func_type* p, int np, double rho[], double sigma[], double * exc, double vrho[], double vsigma[]);

/**
* Evaluates the functional on provided density if functional is MGGA type.
*
* @param[in] p functional to evaluate
* @param[in] np number of grid points (size of rho)
* @param[in] rho electron density
* @param[in] sigma gradient of electron density
* @param[in] lapl laplacian of electron density
* @param[in] tau kinetic energy density
* @param[(in), out] exc energy density. If fp.edens = 0, exc[0] contains energy.
* @param[(in), out] vrho dE/drho
* @param[(in), out] vsigma dE/dsigma
* @param[(in), out] vlapl dE/dlapl
* @param[(in), out] vtau dE/dtau
*/
void nxc_mgga_exc_vxc(nxc_func_type* p, int np, double rho[],double sigma[], double lapl[],
    double tau[], double * exc, double vrho[], double vsigma[], double vlapl[],double vtau[]);

/**
* Check if GPU(cuda) is available
*/
int nxc_cuda_available();

int nxc_func_get_family(nxc_func_type* p);
int nxc_func_get_family_from_path(std::string model);

#endif
