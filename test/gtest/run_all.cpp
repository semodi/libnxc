#include "gtest/gtest.h"
#include "atomicfunc.cpp"

namespace{
TEST(general, loadmodelnopar){
  EXPECT_EQ(0, load_model_nopar());
}
TEST(general, loadmodel){
  EXPECT_EQ(0, load_model());
}
TEST(general, runmodel){
  EXPECT_EQ(0, run_model());
}
}
GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from gtest_main.cc\n");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
