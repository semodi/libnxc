#include "nxc.h"
#include "nxc_func.h"
#include <torch/script.h> // One-stop header.
#include <filesystem>
#include "nxc_mpi.h"
// #include "nxc_mpi.h"


inline void distribute_v(double * v_src, double * v_tar,
   int nouter, int ninner, int offset, double mult){

    for (int is=0; is < nouter; ++is){
      for (int ip=0; ip < ninner; ++ip){
        v_tar[is*ninner+ip] = v_src[(is+offset)*ninner+ip]*mult;
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
}

void LDAFunc::exc_vxc(int np, double rho[], double * exc, double vrho[]){

    // Assuming row-major order to resolve spin, caution when passing from fortran!
    torch::Tensor trho = torch::from_blob(rho, np*nspin, options_dp.requires_grad(true));
    torch::Tensor rho_inp = trho.view({nspin, np});
    if (nspin == NXC_UNPOLARIZED){
        rho_inp = rho_inp.expand({2,-1})*0.5;
    }
    rho_inp = rho_inp.transpose(0,1);
    torch::Tensor texc, Exc;

    texc = model.energy.forward({rho_inp}).toTensor().squeeze();
    Exc = torch::dot(texc, torch::sum(rho_inp, 1));
    Exc.backward();
    torch::Tensor trhograd = trho.grad();
    double *v = trho.grad().data_ptr<double>();
    double *e = texc.data_ptr<double>();
    distribute_v(v, vrho, nspin, np, 0, 1);
    distribute_v(e, exc, 1, np, 0, 1);
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
    double *e = texc.data_ptr<double>();
    distribute_v(vr, vrho, nspin, np, 0, 1);
    distribute_v(vs, vsigma, sigmamult, np, 0, 1);
    distribute_v(e, exc, 1, np, 0, 1);
}
void MGGAFunc::exc_vxc(int np, double rho[], double sigma[], double lapl[],
        double tau[], double * exc, double vrho[], double vsigma[], double vlapl[], double vtau[])
{

    // Assuming row-major order to resolve spin, caution when passing from fortran!
    torch::Tensor trho = torch::from_blob(rho, np*nspin, options_dp.requires_grad(true)).view({nspin,np});
    int sigmamult = (nspin==NXC_UNPOLARIZED) ? 1 : 3; // Three spin channels for sigma if polarized
    torch::Tensor tsigma = torch::from_blob(sigma, np*sigmamult, options_dp.requires_grad(true)).view({sigmamult,np});
    torch::Tensor tlapl = torch::from_blob(lapl, np*nspin, options_dp.requires_grad(true)).view({nspin,np});
    torch::Tensor ttau = torch::from_blob(tau, np*nspin, options_dp.requires_grad(true)).view({nspin,np});

    if (nspin == NXC_UNPOLARIZED){
        trho = trho.expand({2,-1});
        tsigma = trho.expand({3,-1});
        tlapl = trho.expand({2,-1});
        ttau = trho.expand({2,-1});
    }
    torch::Tensor trho0 = torch::cat({trho, tsigma, tlapl, ttau});
    trho0.requires_grad_(true);
    torch::Tensor texc, Exc;
    texc = model.energy.forward({trho0.transpose(0,1)}).toTensor().squeeze();
    Exc = torch::dot(texc,trho.select(0,0)+trho.select(0,1));
    Exc.backward();

    double *v = trho0.grad().data_ptr<double>();
    double *e = texc.data_ptr<double>();
    distribute_v(v, vrho, 2, np, 0, nspin);
    distribute_v(v, vsigma, 3, np, 2, nspin);
    distribute_v(v, vlapl, 2, np, 5, nspin);
    distribute_v(v, vtau, 2, np, 7, nspin);
    distribute_v(e, exc, 1, np, 0, nspin);
}
