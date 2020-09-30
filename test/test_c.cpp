#include <iostream>
#include "nxc.h"
int main(){
    nxc_func_type p;
    nxc_func_init(&p, "test.jit");
    nxc_func_init(&p, "test_agn.jit");
    std::cout << "OK!" << std::endl;
}
