#include "nxc.h"
#include "nxc_func.h"
#include <torch/script.h> // One-stop header.
#include <filesystem>
#define STRINGIZER(arg)     #arg
#define STR_VALUE(arg)      STRINGIZER(arg)
#define NXC_DEFAULTPATH_STRING STR_VALUE(NXC_DEFAULTPATH)

const std::string default_modeldir = NXC_DEFAULTPATH_STRING;

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

    std::string fullname;
    if(std::getenv("NXC_MODELPATH")){
        const char* modeldir;
        modeldir = std::getenv("NXC_MODELPATH");
        fullname = std::string(modeldir) + modelname;
    }else{
        fullname = default_modeldir + modelname;
    }

    std::string path;
    try {
      path =  modelname + "/xc";
      model.energy = torch::jit::load(path);
    }catch (const c10::Error& e) {
      try {
        path =  fullname + "/xc";
        model.energy = torch::jit::load(path);
      }catch (const c10::Error& e) {
        std::cerr << "error loading the model " << path << std::endl;
        throw(e);
      }
      // std::cerr << "error loading the model " << path << std::endl;
      // throw(e);
    }
}

void GridFunc::init(func_param fp, int nspin){
  this->nspin = nspin;
  add = fp.add;
  edens = fp.edens;
  gamma = fp.gamma;
  if (!edens){
    tcell = torch::from_blob(fp.cell, 9, options_dp).view({3,3});
    tgrid = torch::from_blob(fp.grid, 3, options_int);
    torch::Tensor U = torch::zeros({3,3}, options_dp);
    for(int i = 0; i< 3; ++i){
      U.select(1,i) = tcell.select(1,i)/tgrid.select(0,i);
    }
    V_cell = torch::abs(torch::det(U));
  }
  if (fp.cuda) this->to_cuda();
}

void GridFunc::to_cuda(){

  model.energy.to(at::kCUDA);
  device = DEVICE_CUDA;
  std::cout << "libnxc:Using CUDA" << std::endl;

}
void LDAFunc::exc_vxc(int np, double rho[], double * exc, double vrho[]){

    torch::Tensor trho_cud, filter, texc_full;
    // Assuming row-major order to resolve spin, caution when passing from fortran!
    torch::Tensor trho = torch::from_blob(rho, np*nspin, options_dp.requires_grad(true));
    if (device == DEVICE_CUDA){
      trho_cud = trho.to(at::kCUDA);
    }else{
      trho_cud = trho;
    }

    torch::Tensor rho_inp;
    rho_inp = trho_cud.view({nspin, np});
    if (nspin == NXC_UNPOLARIZED){
        rho_inp = rho_inp.expand({2,-1})*0.5;
    }

    torch::Tensor trho0 = rho_inp;
    trho0 = torch::where(trho0 > 0, trho0, torch::zeros_like(trho0));
    filter = torch::sum(trho0.index({torch::tensor({0,1}),"..."}),0) > 1e-7;
    trho0 = trho0.index({"...",filter});
    if (trho0.size(1)==0){
      for(int i=0; i<np;i++){ //TODO: This is not entirely correct
        exc[i] = 0;
        vrho[i] = 0;
      }
    }else{
      torch::Tensor texc, Exc;
      texc_full = torch::zeros_like(rho_inp.select(0,0));
      texc = model.energy.forward({trho0.transpose(0,1)}).toTensor().select(1,0);
      // texc_full = torch::where(filter, texc, texc_full);
      texc_full.index_put_({filter}, texc);
      if (!torch::equal(texc,texc)){
        std::cout << texc.index({filter}) << std::endl;
        throw ("NaN encountered");
      }

      Exc = torch::dot(texc_full, torch::sum(rho_inp, 0));
      Exc.backward();

      if (!torch::equal(trho.grad(),trho.grad())){
        std::cout << trho.grad().index({filter}) << std::endl;
        throw ("NaN encountered");
      }
      texc_full = texc_full.to(at::kCPU);
      Exc = Exc.to(at::kCPU);
      double *vr = trho.grad().data_ptr<double>();
      int npe;
      double *e;
      if (edens){
        e = texc_full.data_ptr<double>();
        npe = np;
      }else{
        e = Exc.data_ptr<double>();
        npe = 1;
      }
      distribute_v(vr, vrho, nspin, np, 0, 1, add);
      distribute_v(e, exc, 1, npe, 0, 1, add);
    }
}

void GGAFunc::exc_vxc(int np, double rho[], double sigma[],
        double * exc, double vrho[], double vsigma[]){


    torch::Tensor trho_cud, tsigma_cud, sigma_aa, sigma_bb, sigma_ab, tsigma, filter, texc_full;
    // Assuming row-major order to resolve spin, caution when passing from fortran!
    torch::Tensor trho = torch::from_blob(rho, np*nspin, options_dp.requires_grad(true));
    int sigmamult;

    if (gamma){
      sigmamult = (nspin==NXC_UNPOLARIZED) ? 3 : 6; // Three spin channels for sigma if polarized
      tsigma = torch::from_blob(sigma, np*sigmamult, options_dp.requires_grad(true));
    }else{
      sigmamult = (nspin==NXC_UNPOLARIZED) ? 1 : 3; // Three spin channels for sigma if polarized
      tsigma = torch::from_blob(sigma, np*sigmamult, options_dp.requires_grad(true));
    }

    if (device == DEVICE_CUDA){
      trho_cud = trho.to(at::kCUDA);
      tsigma_cud = tsigma.to(at::kCUDA);
    }else{
      trho_cud = trho;
      tsigma_cud = tsigma;
    }

    torch::Tensor rho_inp, sigma_inp;
    rho_inp = trho_cud.view({nspin, np});
    sigma_inp = tsigma_cud.view({sigmamult, np});
    if (nspin == NXC_UNPOLARIZED){
        rho_inp = rho_inp.expand({2,-1})*0.5;
        if (gamma){
          sigma_inp = sigma_inp.repeat({2,1})*0.5;
        }else{
          sigma_inp = sigma_inp.expand({3,-1})*0.25;
        }
    }

    if (gamma){
      sigma_aa = sigma_inp.select(0,0)*sigma_inp.select(0,0) +
                 sigma_inp.select(0,1)*sigma_inp.select(0,1) +
                 sigma_inp.select(0,2)*sigma_inp.select(0,2) ;
      sigma_bb = sigma_inp.select(0,3)*sigma_inp.select(0,3) +
                 sigma_inp.select(0,4)*sigma_inp.select(0,4) +
                 sigma_inp.select(0,5)*sigma_inp.select(0,5) ;
      sigma_ab = sigma_inp.select(0,0)*sigma_inp.select(0,3) +
                 sigma_inp.select(0,1)*sigma_inp.select(0,4) +
                 sigma_inp.select(0,2)*sigma_inp.select(0,5) ;

      sigma_inp = torch::cat({sigma_aa.unsqueeze(0), sigma_ab.unsqueeze(0), sigma_bb.unsqueeze(0)});
    }

    torch::Tensor trho0 = torch::cat({rho_inp, sigma_inp});
    trho0 = torch::where(trho0 > 0, trho0, torch::zeros_like(trho0));
    filter = torch::sum(trho0.index({torch::tensor({0,1}),"..."}),0) > 1e-10;
    trho0 = trho0.index({"...",filter});
    bool padded_trho0 = false;

    if (trho0.size(1)==1){
      trho0 = trho0.repeat({1,2});
      padded_trho0 = true;
    }
    if (trho0.size(1)==0){
      for(int i=0; i<np;i++){ //TODO: This is not entirely correct
        exc[i] = 0;
        vrho[i] = 0;
        vsigma[i] = 0;
      }
    }else{
      torch::Tensor texc, Exc;
      texc_full = torch::zeros_like(rho_inp.select(0,0));
      texc = model.energy.forward({trho0.transpose(0,1)}).toTensor().select(1,0);
      if (padded_trho0)
        texc = texc.index({"...",0});
      // texc_full = torch::where(filter, texc, texc_full);
      texc_full.index_put_({filter}, texc);
      if (!torch::equal(texc,texc)){
        std::cout << texc.index({filter}) << std::endl;
        throw ("NaN encountered");
      }
      Exc = torch::dot(texc_full, torch::sum(rho_inp, 0));
      Exc.backward();

      if (!torch::equal(trho.grad(),trho.grad())){
        std::cout << trho.grad().index({filter}) << std::endl;
        throw ("NaN encountered");
      }
      texc_full = texc_full.to(at::kCPU);
      Exc = Exc.to(at::kCPU);
      double *vr = trho.grad().data_ptr<double>();
      double *vs = tsigma.grad().data_ptr<double>();
      int npe;
      double *e;
      if (edens){
        e = texc_full.data_ptr<double>();
        npe = np;
      }else{
        e = Exc.data_ptr<double>();
        npe = 1;
      }
      distribute_v(vr, vrho, nspin, np, 0, 1, add);
      distribute_v(vs, vsigma, sigmamult, np, 0, 1, add);
      distribute_v(e, exc, 1, npe, 0, 1, add);
    }
}
void MGGAFunc::exc_vxc(int np, double rho[], double sigma[], double lapl[],
        double tau[], double * exc, double vrho[], double vsigma[], double vlapl[], double vtau[])
{
    torch::Tensor trho_cud, tsigma_cud, sigma_aa, sigma_bb,
      sigma_ab, tsigma, filter, texc_full, tlapl_cud, ttau_cud;
    // Assuming row-major order to resolve spin, caution when passing from fortran!
    torch::Tensor trho = torch::from_blob(rho, np*nspin, options_dp.requires_grad(true));
    int sigmamult;

    if (gamma){
      sigmamult = (nspin==NXC_UNPOLARIZED) ? 3 : 6; // Three spin channels for sigma if polarized
      tsigma = torch::from_blob(sigma, np*sigmamult, options_dp.requires_grad(true));
    }else{
      sigmamult = (nspin==NXC_UNPOLARIZED) ? 1 : 3; // Three spin channels for sigma if polarized
      tsigma = torch::from_blob(sigma, np*sigmamult, options_dp.requires_grad(true));
    }
    torch::Tensor tlapl = torch::from_blob(lapl, np*nspin, options_dp.requires_grad(true));
    torch::Tensor ttau = torch::from_blob(tau, np*nspin, options_dp.requires_grad(true));

    if (device == DEVICE_CUDA){
      trho_cud = trho.to(at::kCUDA);
      tsigma_cud = tsigma.to(at::kCUDA);
      tlapl_cud = tlapl.to(at::kCUDA);
      ttau_cud = ttau.to(at::kCUDA);
    }else{
      trho_cud = trho;
      tsigma_cud = tsigma;
      tlapl_cud = tlapl;
      ttau_cud = ttau;
    }

    torch::Tensor rho_inp, sigma_inp, lapl_inp, tau_inp;
    rho_inp = trho_cud.view({nspin, np});
    sigma_inp = tsigma_cud.view({sigmamult, np});
    lapl_inp = tlapl_cud.view({nspin, np});
    tau_inp = ttau_cud.view({nspin, np});

    if (nspin == NXC_UNPOLARIZED){
        rho_inp = rho_inp.expand({2,-1})*0.5;
        if (gamma){
          sigma_inp = sigma_inp.repeat({2,1})*0.5;
        }else{
          sigma_inp = sigma_inp.expand({3,-1})*0.25;
        }
        tau_inp = tau_inp.expand({2,-1})*0.5;
        lapl_inp = lapl_inp.expand({2,-1})*0.5;
    }

    if (gamma){
      sigma_aa = sigma_inp.select(0,0)*sigma_inp.select(0,0) +
                 sigma_inp.select(0,1)*sigma_inp.select(0,1) +
                 sigma_inp.select(0,2)*sigma_inp.select(0,2) ;
      sigma_bb = sigma_inp.select(0,3)*sigma_inp.select(0,3) +
                 sigma_inp.select(0,4)*sigma_inp.select(0,4) +
                 sigma_inp.select(0,5)*sigma_inp.select(0,5) ;
      sigma_ab = sigma_inp.select(0,0)*sigma_inp.select(0,3) +
                 sigma_inp.select(0,1)*sigma_inp.select(0,4) +
                 sigma_inp.select(0,2)*sigma_inp.select(0,5) ;

      sigma_inp = torch::cat({sigma_aa.unsqueeze(0), sigma_ab.unsqueeze(0), sigma_bb.unsqueeze(0)});
    }

    torch::Tensor trho0 = torch::cat({rho_inp, sigma_inp, lapl_inp, tau_inp});
    trho0 = torch::where(trho0 > 0, trho0, torch::zeros_like(trho0));
    filter = torch::sum(trho0.index({torch::tensor({0,1}),"..."}),0) > 1e-10;
    trho0 = trho0.index({"...",filter});
    if (trho0.size(1)==0){
      for(int i=0; i<np;i++){ //TODO: This is not entirely correct
        exc[i] = 0;
        vrho[i] = 0;
        vsigma[i] = 0;
        vlapl[i]=0;
        vtau[i]=0;
      }
    }else{
      torch::Tensor texc, Exc;
      texc_full = torch::zeros_like(rho_inp.select(0,0));
      texc = model.energy.forward({trho0.transpose(0,1)}).toTensor().select(1,0);
      // texc_full = torch::where(filter, texc, texc_full);
      texc_full.index_put_({filter}, texc);
      if (!torch::equal(texc,texc)){
        std::cout << texc.index({filter}) << std::endl;
        throw ("NaN encountered");
      }

      Exc = torch::dot(texc_full, torch::sum(rho_inp, 0));
      Exc.backward();

      if (!torch::equal(trho.grad(),trho.grad())){
        std::cout << trho.grad().index({filter}) << std::endl;
        throw ("NaN encountered");
      }
      texc_full = texc_full.to(at::kCPU);
      Exc = Exc.to(at::kCPU);
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
}
