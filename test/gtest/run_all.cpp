#include "gtest/gtest.h"
#include "atomic.h"
#include "grid.h"

namespace{
TEST(atomic, loadmodelnopar){
  // Load model without any functional parameters
  EXPECT_EQ(0, load_model_nopar());
}
TEST(atomic, loadmodel){
  // Load model with functional parameters
  EXPECT_EQ(0, load_model());
}
TEST(atomic, runmodel){
  // Run the test model on synthetic data
  int myBox[6] = {0, 9, 0, 9, 0, 9};
  EXPECT_LT(abs(0.00328425-run_model(myBox)),1e-8);
}
TEST(atomic, testbox){
  // Check myBox consistency (important if using MPI)
  int myBox1[6] = {0, 4, 0, 4, 0, 4};
  int myBox2[6] = {5, 9, 5, 9, 5, 9};
  int myBox3[6] = {3, 7, 3, 7, 3, 7};
  EXPECT_LT(abs(run_model(myBox1)-run_model(myBox2)),1e-8);
  EXPECT_GT(abs(run_model(myBox3)-run_model(myBox2)),1e-8);
}

TEST(grid, loadhmmodels){
    load_hm_models();
}
TEST(grid, testhmlda){
    double vrho_up[100];
    double exc_up[100];
    double vrho_p[200];
    double exc_p[100];
    test_hm_lda(vrho_up, exc_up, vrho_p, exc_p);

    // Compare spin-polarized to unpolarized results and check
    // for consistency
    for( int i=0;i<100; ++i){
      EXPECT_LT(abs(exc_up[i]-exc_p[i]),1e-8);
    }

    for( int i=0;i<100; ++i){
      EXPECT_LT(abs(vrho_up[i]-vrho_p[i]),1e-8);
    }
    for( int i=0;i<100; ++i){
      EXPECT_LT(abs(vrho_up[i]-vrho_p[i+100]),1e-8);
    }
}

}
GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from gtest_main.cc\n");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
