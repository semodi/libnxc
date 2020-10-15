#include "nxc.h"
#include "nxc_func.h"
#include <torch/script.h> // One-stop header.
#include <filesystem>
// #include "nxc_mpi.h"


inline void distribute_v(double * v_src, double * v_tar,
   int nouter, int ninner, int offset, double mult, bool add){

   if(add){
    for (int is=0; is < nouter; ++is){
      for (int ip=0; ip < ninner; ++ip){
        v_tar[is*ninner+ip] += v_src[(is+offset)*ninner+ip]*mult;
      }
    }
   }else{
    for (int is=0; is < nouter; ++is){
      for (int ip=0; ip < ninner; ++ip){
        v_tar[is*ninner+ip] = v_src[(is+offset)*ninner+ip]*mult;
      }
    }
   }
}
GridFunc::GridFunc(std::string modelname){

    const char* modeldir;
    if(std::getenv("NXC_MODELPATH")){
        modeldir = std::getenv("NXC_MODELPATH");
    }else{
        std::cout << "Need to set NXC_MODELPATH environment variable!" << std::endl;
        throw ("Need to set NXC_MODELPATH environment variable!");
    }

    modelname = std::string(modeldir) + modelname;
    std::string path;
    try {
      path =  modelname + "/xc";
      model.energy = torch::jit::load(path);
    }catch (const c10::Error& e) {
      std::cerr << "error loading the model " << path << std::endl;
      throw(e);
    }
}

void GridFunc::init(func_param fp, int nspin){
  this->nspin = nspin;
  add = fp.add;
  edens = fp.edens;
  if (!edens){
    tcell = torch::from_blob(fp.cell, 9, options_dp).view({3,3});
    tgrid = torch::from_blob(fp.grid, 3, options_int);
    torch::Tensor U = torch::zeros({3,3}, options_dp);
    for(int i = 0; i< 3; ++i){
      U.select(1,i) = tcell.select(1,i)/tgrid.select(0,i);
    }
    V_cell = torch::abs(torch::det(U));
  }
}

void LDAFunc::exc_vxc(int np, double rho[], double * exc, double vrho[]){

    // Assuming row-major order to resolve spin, caution when passing from fortran!
    torch::Tensor trho = torch::from_blob(rho, np*nspin, options_dp.requires_grad(true));

    torch::Tensor trho0;
    trho0 = trho.view({nspin, np});
    trho0 = trho0.index({trho0 > 1e-7});

    if (nspin == NXC_UNPOLARIZED){
        trho0 = trho0.expand({2,-1})*0.5;
    }

    torch::Tensor texc, Exc;
    texc = model.energy.forward({trho0.transpose(0,1)}).toTensor().squeeze();
    Exc = torch::dot(texc, torch::sum(trho0, 0));
    Exc.backward();

    double *vr = trho.grad().data_ptr<double>();
    distribute_v(vr, vrho, nspin, np, 0, 1, add);
    int npe;
    double *e;
    if (edens){
      e = texc.data_ptr<double>();
      npe = np;
    }else{
      e = Exc.data_ptr<double>();
      e[0] = e[0]*(V_cell.data_ptr<double>()[0]);
      npe = 1;
    }
    distribute_v(e, exc, 1, npe, 0, 1, add);
}

void GGAFunc::exc_vxc(int np, double rho[], double sigma[],
        double * exc, double vrho[], double vsigma[]){

    // Assuming row-major order to resolve spin, caution when passing from fortran!
    torch::Tensor trho = torch::from_blob(rho, np*nspin, options_dp.requires_grad(true));
    int sigmamult = (nspin==NXC_UNPOLARIZED) ? 1 : 3; // Three spin channels for sigma if polarized
    torch::Tensor tsigma = torch::from_blob(sigma, np*sigmamult, options_dp.requires_grad(true));

    torch::Tensor rho_inp,sigma_inp;
    rho_inp = trho.view({nspin, np});
    sigma_inp = tsigma.view({sigmamult, np});

    if (nspin == NXC_UNPOLARIZED){
        rho_inp = rho_inp.expand({2,-1})*0.5;
        sigma_inp = sigma_inp.expand({3,-1})*0.25;
    }

    torch::Tensor trho0 = torch::cat({rho_inp, sigma_inp});
    torch::Tensor texc, Exc;
    texc = model.energy.forward({trho0.transpose(0,1)}).toTensor().squeeze();
    Exc = torch::dot(texc, torch::sum(rho_inp, 0));
    Exc.backward();

    double *vr = trho.grad().data_ptr<double>();
    double *vs = tsigma.grad().data_ptr<double>();
    int npe;
    double *e;
    if (edens){
      e = texc.data_ptr<double>();
      npe = np;
    }else{
      e = Exc.data_ptr<double>();
      npe = 1;
    }
    distribute_v(vr, vrho, nspin, np, 0, 1, add);
    distribute_v(vs, vsigma, sigmamult, np, 0, 1, add);
    distribute_v(e, exc, 1, npe, 0, 1, add);
}
void MGGAFunc::exc_vxc(int np, double rho[], double sigma[], double lapl[],
        double tau[], double * exc, double vrho[], double vsigma[], double vlapl[], double vtau[])
{

    // Assuming row-major order to resolve spin, caution when passing from fortran!
    torch::Tensor trho = torch::from_blob(rho, np*nspin, options_dp.requires_grad(true));
    int sigmamult = (nspin==NXC_UNPOLARIZED) ? 1 : 3; // Three spin channels for sigma if polarized
    torch::Tensor tsigma = torch::from_blob(sigma, np*sigmamult, options_dp.requires_grad(true));
    torch::Tensor tlapl = torch::from_blob(lapl, np*nspin, options_dp.requires_grad(true));
    torch::Tensor ttau = torch::from_blob(tau, np*nspin, options_dp.requires_grad(true));

    torch::Tensor rho_inp, sigma_inp, lapl_inp, tau_inp;
    rho_inp = trho.view({nspin, np});
    sigma_inp = tsigma.view({sigmamult, np});
    lapl_inp = tlapl.view({nspin, np});
    tau_inp = ttau.view({nspin, np});

    if (nspin == NXC_UNPOLARIZED){
        rho_inp = rho_inp.expand({2,-1})*0.5;
        sigma_inp = sigma_inp.expand({3,-1})*0.25;
        tau_inp = tau_inp.expand({2,-1})*0.5;
        lapl_inp = lapl_inp.expand({2,-1})*0.5;
    }

    torch::Tensor trho0 = torch::cat({rho_inp, sigma_inp, lapl_inp, tau_inp});
    torch::Tensor texc, Exc;
    texc = model.energy.forward({trho0.transpose(0,1)}).toTensor().squeeze();
    Exc = torch::dot(texc, torch::sum(rho_inp, 0));
    Exc.backward();

    double *vr = trho.grad().data_ptr<double>();
    double *vs = tsigma.grad().data_ptr<double>();
    double *vl = tlapl.grad().data_ptr<double>();
    double *vt = ttau.grad().data_ptr<double>();
    int npe;
    double *e;
    if (edens){
      e = texc.data_ptr<double>();
      npe = np;
    }else{
      e = Exc.data_ptr<double>();
      npe = 1;
    }
    distribute_v(vr, vrho, nspin, np, 0, 1, add);
    distribute_v(vt, vtau, nspin, np, 0, 1, add);
    distribute_v(vl, vlapl, nspin, np, 0, 1, add);
    distribute_v(vs, vsigma, sigmamult, np, 0, 1, add);
    distribute_v(e, exc, 1, npe, 0, 1, add);
}
