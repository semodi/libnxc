#ifndef NXC_FUNC_H
#define NXC_FUNC_H
#include "nxc.h"
#include <torch/script.h> // One-stop header.
// #include "nxc_mpi.h"

const double RY = 13.605693012183622;
const double HARTREE = 27.211386024367243;
const auto options_dp = torch::TensorOptions().dtype(torch::kFloat64);
const auto options_int = torch::TensorOptions().dtype(torch::kInt);

const int LDA_TYPE=0;
const int GGA_TYPE=1;
const int MGGA_TYPE=2;
const int ATOMIC_TYPE=4;

const int DEVICE_CUDA=1;
const int DEVICE_CPU=0;

const std::vector<std::string> model_names = {
  "LDA_HM",
  "GGA_HM",
  "MGGA_HM",
  "GGA_PBE",
  "GGA_X_PBE",
  "GGA_KSR",
  "GGA_C_PBE"
};
const std::vector<int> model_types = {
  LDA_TYPE,
  GGA_TYPE,
  MGGA_TYPE,
  GGA_TYPE,
  GGA_TYPE,
  GGA_TYPE,
  GGA_TYPE
};

struct modules{
    torch::jit::script::Module basis;
    torch::jit::script::Module projector;
    torch::jit::script::Module energy;
};


class GridFunc : public NXCFunc{

  public:
    GridFunc(std::string modelname);
    void init(func_param fp, int nspin);
    void exc_vxc_fs(int np, double rho[], double * exc, double vrho[], double forces[], double stress[]){
      exc_vxc(np, rho, exc, vrho);
    }
    void exc_vxc_fs(int np, double rho[],  double sigma[],
       double * exc, double vrho[], double vsigma[], double forces[], double stress[]){
        exc_vxc(np, rho, sigma, exc, vrho, vsigma);
    }
    void exc_vxc_fs(int np, double rho[], double sigma[], double lapl[],
        double tau[], double * exc, double vrho[], double vsigma[], double vlapl[], double vtau[], double forces[], double stress[]){
          exc_vxc(np, rho, sigma, lapl, tau, exc, vrho, vsigma, vlapl, vtau);
    }
    void to_cuda();

  protected:
    modules model;
    bool edens;
    bool add;
    bool gamma;
    torch::Tensor tcell, tgrid, V_cell;

};

class LDAFunc : public GridFunc{

  public:
    // LDAFunc(std::string modelname) : GridFunc(modelname){};
    using GridFunc::GridFunc;
    void exc_vxc(int np, double rho[], double * exc, double vrho[]);
    int get_family(){return LDA_TYPE;};
};
class GGAFunc : public GridFunc{

  public:
    // GGAFunc(std::string modelname)){};
    using GridFunc::GridFunc;
    void exc_vxc(int np, double rho[],  double sigma[],
       double * exc, double vrho[], double vsigma[]);
    int get_family(){return GGA_TYPE;};
};
class MGGAFunc : public GridFunc{

  public:
    using GridFunc::GridFunc;
    // MGGAFunc(std::string modelname){};
    void exc_vxc(int np, double rho[], double sigma[], double lapl[],
        double tau[], double * exc, double vrho[], double vsigma[], double vlapl[], double vtau[]);
    int get_family(){return MGGA_TYPE;};
};

class AtomicFunc : public NXCFunc {

  public:
    AtomicFunc(std::string modeldir);
    void init(func_param fp, int nspin);
    void exc_vxc(int np, double rho[], double * exc, double vrho[]);
    void exc_vxc_fs(int np, double rho[], double * exc, double vrho[],
                          double forces[], double stress[]);
    void to_cuda();
    int get_family(){return ATOMIC_TYPE;};


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
    bool add;
    bool gamma;

    void build_basis();
    void get_descriptors(torch::Tensor &rho, torch::Tensor* descr);
};


#endif
