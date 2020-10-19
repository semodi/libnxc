#include <iostream>
#include "nxc.h"

void load_hm_models();
void test_hm_lda(double vrho_up[],double exc_up[], double vrho_p[], double exc_p[], bool cuda);
void test_hm_gga(double vrho_up[], double vsigma_up[], double exc_up[],
                 double vrho_p[], double vsigma_p[], double exc_p[], bool cuda);
void test_hm_mgga(double vrho_up[], double vsigma_up[], double vlapl_up[], double vtau_up[], double exc_up[],
                 double vrho_p[], double vsigma_p[], double vlapl_p[], double vtau_p[], double exc_p[], bool cuda);
