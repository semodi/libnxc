#ifndef nxc_func_h
#define nxc_func_h
#include "nxc.h"
#include <torch/script.h> // One-stop header.

const double RY = 13.605693012183622;
const double HARTREE = 27.211386024367243;
const auto options_dp = torch::TensorOptions().dtype(torch::kFloat64);
const auto options_int = torch::TensorOptions().dtype(torch::kInt);

struct modules{
    torch::jit::script::Module basis;
    torch::jit::script::Module projector;
    torch::jit::script::Module energy;
};

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

class AtomicFunc : public NXCFunc {

public:
  AtomicFunc(std::string modeldir);
  void init(func_param fp);
  void exc_vxc(int np, double rho[], double * exc, double vrho[]);
  void exc_vxc_fs(int np, double rho[], double * exc, double vrho[],
                        double forces[], double stress[]);



private:
  func_param param;
  std::map<std::string, modules> all_mods;
  std::vector<at::Tensor> all_rads;
  std::vector<at::Tensor> all_angs;
  std::vector<at::Tensor> all_boxes;
  bool models_loaded = false;
  bool any_onrank = true;
  torch::Tensor U, V_cell , V_ucell, tcell, tgrid, tgrid_d, tpos, tpos_flat, my_box;
  bool self_consistent = true;
  bool last_step = false;
  bool agnostic = false;
  std::vector<int>  isa_glob;
  std::vector<std::string> symbols;
  torch::Tensor epsilon = torch::zeros({3,3}, options_dp);
  int * grid_glob;
  int box_dim[3];
  bool edens;

  void build_basis();

};

struct nxc_func_type{
  std::shared_ptr <NXCFunc> func;
};

std::shared_ptr<AtomicFunc> get_functional(std::string modeldir);
#endif
