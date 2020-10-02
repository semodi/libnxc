#include "gtest/gtest.h"
#include "atomic.h"

namespace{
TEST(general, loadmodelnopar){
  // Load model without any functional parameters
  EXPECT_EQ(0, load_model_nopar());
}
TEST(general, loadmodel){
  // Load model with functional parameters
  EXPECT_EQ(0, load_model());
}
TEST(general, runmodel){
  // Run the test model on synthetic data
  int myBox[6] = {0, 9, 0, 9, 0, 9};
  EXPECT_LT(abs(0.00328425-run_model(myBox)),0.0000001);
}
TEST(general, testbox){
  // Check myBox consistency (important if using MPI)
  int myBox1[6] = {0, 4, 0, 4, 0, 4};
  int myBox2[6] = {5, 9, 5, 9, 5, 9};
  int myBox3[6] = {3, 7, 3, 7, 3, 7};
  EXPECT_LT(abs(run_model(myBox1)-run_model(myBox2)),0.0000001);
  EXPECT_GT(abs(run_model(myBox3)-run_model(myBox2)),0.0000001);
}
}
GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from gtest_main.cc\n");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
