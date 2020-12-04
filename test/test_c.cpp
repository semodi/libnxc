#include <iostream>
#include "nxc.h"

char GREEN[8] = "\033[92m";
char END[8]= "\033[0m";

int main(){
    nxc_func_type p;
    nxc_func_init(&p, "test.jit");
    nxc_func_init(&p, "GGA_PBE");
    std::cout << GREEN << " OK!" << END <<  std::endl;
}
