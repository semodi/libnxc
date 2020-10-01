#include "gtest/gtest.h"
#include "load_model.cpp"

namespace{
TEST(general, loadmodelnopar){
  EXPECT_EQ(0, load_model_nopar());
}
TEST(general, loadmodel){
  EXPECT_EQ(0, load_model());
}
}
GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from gtest_main.cc\n");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
