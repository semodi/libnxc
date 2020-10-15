#include <iostream>
#include "nxc.h"

void load_hm_models();
void test_hm_lda(double vrho_up[],double exc_up[], double vrho_p[], double exc_p[]);
void test_hm_gga(double vrho_up[], double vsigma_up[], double exc_up[],
                 double vrho_p[], double vsigma_p[], double exc_p[]);
