#include "cuda.h"

int test_cuda(){

  return cuda_available();
}

int send_model(){

    nxc_func_type p;
    func_param fp;

    nxc_func_init(&p,"HM_LDA", fp);
    p.func->to_cuda();
    return 0;

}
