#include "cuda.h"

int test_cuda(){

  return nxc_cuda_available();
}

int send_model(){

    nxc_func_type p;
    func_param fp;

    nxc_func_init(&p,"LDA_HM", fp);
    p.func->to_cuda();
    return 0;

}
