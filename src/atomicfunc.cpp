#include "nxc_func.h"
#include "nxc.h"
#include <torch/script.h> // One-stop header.
#include <filesystem>
#include "nxc_mpi.h"
// #include "nxc_mpi.h"



AtomicFunc::AtomicFunc(std::string modeldir){

    std::string xc_str = "xc_";
    std::vector<std::string> model_symbols;
    int ns = 0;
    for (const auto & entry : std::filesystem::directory_iterator(modeldir)){
        std::string file = entry.path().filename();
        if(file.find(xc_str) != std::string::npos){
          ns++;
          model_symbols.push_back(file.substr(file.find(xc_str) + xc_str.size()));
        }
        else if(file.find("NO_SC") != std::string::npos){
          self_consistent=false;
        }else if(file.find("AGN") != std::string::npos){
          agnostic=true;
        }
    }
    std::string path;
    std::string symbol;
    for (int i = 0; i < ns; i += 1){
      try {
        if (model_symbols[i].size() == 1){
            symbol = model_symbols[i] + " ";
        }else{
            symbol = model_symbols[i];
        }
        path =  modeldir + "/basis_" + model_symbols[i];
        all_mods[symbol].basis = torch::jit::load(path);
        path =  modeldir + "/projector_" + model_symbols[i];
        all_mods[symbol].projector = torch::jit::load(path);
        path =  modeldir + "/xc_" + model_symbols[i];
        all_mods[symbol].energy = torch::jit::load(path);

      }catch (const c10::Error& e) {
        std::cerr << "error loading the model " << path << std::endl;
      }
    }
}

void AtomicFunc::init(func_param fp, int nspin){

  edens = fp.edens;
  int npos = (fp.nua) * 3;
  tpos_flat = torch::from_blob(fp.pos, npos, options_dp).clone();
  tpos_flat += 1e-7;
  tpos = tpos_flat.view({fp.nua,3});
  tcell = torch::from_blob(fp.cell, 9, options_dp).view({3,3});
  tgrid = torch::from_blob(fp.grid, 3, options_int);
  tgrid_d = tgrid.clone().to(torch::kFloat64);
  U = torch::zeros({3,3}, options_dp);
  my_box = torch::from_blob(fp.myBox,6, options_int).view({3,2}).clone();


  for(int i = 0; i< 3; ++i){
    my_box[i][1] += 1;
    box_dim[i] = *((my_box[i][1] - my_box[i][0]).data_ptr<int>());
    U.select(1,i) = tcell.select(1,i)/tgrid.select(0,i);
  }

  isa_glob.resize(fp.nua);
  isa_glob.assign(fp.isa, fp.isa+(fp.nua));
  grid_glob = fp.grid;
  V_cell = torch::abs(torch::det(U));
  V_ucell = torch::abs(torch::det(tcell));
  all_rads.resize(fp.nua);
  all_angs.resize(fp.nua);
  all_boxes.resize(fp.nua);
  symbols.resize(fp.ns);
  for(int i = 0; i<2*fp.ns; i+=2){
    std::string symbol_string(fp.symbols + i,2);
    if (agnostic){
      symbol_string = "X ";
    }
    symbols[i/2] = symbol_string;
  }
  this->build_basis();
}

void AtomicFunc::get_descriptors(torch::Tensor &rho, torch::Tensor *descr){

    int ns = symbols.size();
    int nua = tpos.size(0);
    std::vector<bool> on_rank;
    on_rank.resize(nua);

    torch::Tensor this_pos, rad, ang, box, descr_glob[nua];
    torch::Tensor scaler = torch::eye({3}) + epsilon;
    any_onrank=false;
    // Get descriptors
    for(int is = 0; is < ns; ++is){
      for(int ia = 0; ia < nua; ++ia){
        int struct_idx = isa_glob[ia] - 1;
        if(is != struct_idx) continue;
        this_pos = tpos.select(0,ia);
        rad = all_rads[ia];
        ang = all_angs[ia];
        box = all_boxes[ia];
        if (rad.size(-1) == 0)
        {
          descr[ia] = torch::zeros({0}, options_dp);
          on_rank[ia] = false;
        }
        else
        {
          descr[ia] = all_mods[symbols[struct_idx]].projector.forward({rho, torch::mv(scaler,this_pos), torch::mm(tcell,scaler),
              tgrid_d,  rad, ang, box}).toTensor();
          on_rank[ia] = true;
          any_onrank = true;
        }
      }
    }

  #ifdef MPI
    // Reduce descriptors over ranks
    for(int ia = 0; ia < nua; ++ia){
        int dsize = descr[ia].size(0);
        int dsize_max;
        int one=1;
        mpi_allreduce_(&dsize, &dsize_max,&one,&MPI_INTEGER,&MPI_MAX, &MPI_COMM_WORLD, &mpierror);
        if (dsize == 0) descr[ia] = torch::zeros({dsize_max},options_dp);

        std::vector<double> descr_glob_ia;
        descr_glob_ia.resize(dsize_max);

        // double descr_glob_ia;
        double *descr_data = descr[ia].data_ptr<double>();
        mpi_allreduce_(descr_data, descr_glob_ia.data(),
          &dsize_max, &MPI_DOUBLE, &MPI_SUM, &MPI_COMM_WORLD, &mpierror);
        for(int u =0; u < dsize_max; ++u){
          descr_glob_ia[u] -= descr_data[u];
        }
        torch::Tensor t_descr_glob = torch::from_blob(descr_glob_ia.data(),dsize_max,
         options_dp);
        descr[ia] += t_descr_glob;
    }
  #endif

}
void AtomicFunc::exc_vxc(int np, double rho[], double exc[], double vrho[]){
  if (self_consistent || last_step){
    int ns = symbols.size();
    int nua = tpos.size(0);
    torch::Tensor trho= torch::from_blob(rho, np, options_dp.requires_grad(true));
    torch::Tensor trho_view = trho.view({box_dim[2], box_dim[1], box_dim[0]}).transpose(0,2);
    torch::Tensor E = torch::zeros({1}, options_dp);
    torch::Tensor descr[nua], e;

    this->get_descriptors(trho_view, descr);
    // Get energy
    for(int is = 0; is < ns; ++is){
      for(int ia = 0; ia < nua; ++ia)
      {
          int struct_idx = isa_glob[ia] - 1;
          if(is != struct_idx) continue;
          if (descr[ia].size(0) == 0) continue;
          e = all_mods[symbols[struct_idx]].energy.forward({descr[ia].unsqueeze(0)}).toTensor();
          E += e;
      }
    }
    E = E/HARTREE;
    if (any_onrank && self_consistent){
        E.backward();

        torch::Tensor Vnxc = trho.grad()/V_cell;
        double *grad =  Vnxc.data_ptr<double>();
        for(int i = 0; i < np; ++i){
          vrho[i] += grad[i];
        }
    }
    if (edens){
      double grid_factor  = (double(np))/double(*(torch::prod(tgrid).data_ptr<long>()));
      torch::Tensor qtot = torch::sum(trho)*V_cell;
      torch::Tensor t_exc =E/qtot*torch::ones_like(trho)*grid_factor;
      double *E_data = t_exc.data_ptr<double>();
      for(int i=0; i < np; ++i){
        exc[i] += E_data[i];
      }
    }else{
      double *E_data = E.data_ptr<double>();
      exc[0] += E_data[0];
    }
  }
}


void AtomicFunc::exc_vxc_fs(int np, double rho[], double exc[], double vrho[],
                        double forces[], double stress[]){

  int nua = tpos.size(0);
  if (self_consistent){
    tpos_flat.requires_grad_(true);
    epsilon.requires_grad_(true);
  }
  tpos = tpos_flat.view({nua,3});

  last_step=true;
  build_basis();
  this->exc_vxc(np, rho, exc, vrho);
  last_step=false;

  torch::Tensor force_corr, stress_corr;
  if (any_onrank && self_consistent){
    force_corr = tpos_flat.grad();
    stress_corr = epsilon.grad()/V_ucell;
  }else{
    force_corr = torch::zeros({nua*3}, options_dp);
    stress_corr = torch::zeros({3, 3}, options_dp);
  }
  stress_corr = .5*(stress_corr + stress_corr.transpose(0,1));

  double *grad =  force_corr.data_ptr<double>();
  for(int i = 0; i < 3*nua; ++i){
    forces[i] -= grad[i];
  }

  stress_corr = stress_corr.contiguous();
  double *stressgrad = stress_corr.data_ptr<double>();

#ifdef MPI
  double stressgrad_glob[9];
  int nine = 9;
    mpi_allreduce_(stressgrad, stressgrad_glob,
      &nine, &MPI_DOUBLE, &MPI_SUM, &MPI_COMM_WORLD, &mpierror);
    for (int is = 0; is < 9; ++is){
      stress[is] += stressgrad_glob[is];
    }
#else
    for (int is = 0; is < 9; ++is){
      stress[is] += stressgrad[is];
    }
#endif

  tpos_flat.requires_grad_(false);
  tpos = tpos_flat.view({nua,3});

}


void AtomicFunc::build_basis(){
  int ns = symbols.size();
  int nua = tpos.size(0);
  at::Tensor this_pos, rad, ang;

  torch::Tensor scaler = torch::eye({3}) + epsilon;
  for(int is = 0; is < ns; ++is){
    for(int ia = 0; ia < nua; ++ia){
      int struct_idx = isa_glob[ia] - 1;
      if(is != struct_idx) continue;
      this_pos = tpos.select(0,ia);
      auto output = all_mods[symbols[struct_idx]].basis.forward({torch::mv(scaler, this_pos), torch::mm(tcell, scaler),
                                                                tgrid_d,  my_box});
      all_rads[ia] = output.toTuple()->elements()[0].toTensor();
      all_angs[ia] = output.toTuple()->elements()[1].toTensor();
      all_boxes[ia] = output.toTuple()->elements()[2].toTensor();
    }
  }

}
