#include "gtest/gtest.h"
#include "atomic.h"
#include "grid.h"
#include "cuda.h"
#ifdef LIBXC
#include "libxc.h"
#endif
const double TOL = 1e-6;
namespace{

#ifdef CUDA
TEST(cuda, cudaavail){
  EXPECT_EQ(1, test_cuda());
}
TEST(cuda, sendmodel){
  EXPECT_EQ(0, send_model());
}
#endif

#ifdef LIBXC
TEST(libxc, libxcpbe){
  double exc1[5];
  double vrho1[5];
  double vsigma1[5];
  double vtau1[5];
  double exc2[5];
  double vrho2[5];
  double vsigma2[5];
  double vtau2[5];
  const int nfunc = 3;
  int func1[nfunc] = {801, 802, 806};
  int func2[nfunc] = {101, 130, 101};

  for (int j=0; j<nfunc;++j){
    std::cout << "Comparing" << std::endl;
    test_func(func1[j], exc1, vrho1, vsigma1, vtau1);
    std::cout << "to" << std::endl;
    test_func(func2[j], exc2, vrho2, vsigma2, vtau2);
    for (int i=0; i<5;++i){
      EXPECT_LT(abs(exc1[i]-exc2[i]),1e-3);
      EXPECT_LT(abs(vrho1[i]-vrho2[i]),1e-3);
      EXPECT_LT(abs(vsigma1[i]-vsigma2[i]),1e-3);
    }
    std::cout << "==========" << std::endl;
  }
}
#endif

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
  for (int cuda=0;cuda<test_cuda()+1;++cuda){
    EXPECT_LT(abs(0.0314854-run_model(myBox, cuda)),TOL);
  }
}
TEST(atomic, testbox){
  // Check myBox consistency (important if using MPI)
  int myBox1[6] = {0, 4, 0, 4, 0, 4};
  int myBox2[6] = {5, 9, 5, 9, 5, 9};
  int myBox3[6] = {3, 7, 3, 7, 3, 7};
  for (int cuda=0;cuda<test_cuda()+1;++cuda){
    EXPECT_LT(abs(run_model(myBox1, cuda)-run_model(myBox2, cuda)),TOL);
    EXPECT_GT(abs(run_model(myBox3, cuda)-run_model(myBox2, cuda)),TOL);
  }
}

TEST(grid, loadhmmodels){
    load_hm_models();
}
TEST(grid, testhmlda){
    // TODO: parametrize tests with CUDA option
    int cuda=0;
    for (cuda=0;cuda<test_cuda()+1;++cuda){
      double vrho_up[70];
      double exc_up[70];
      double vrho_p[140];
      double exc_p[140];
      test_hm_lda(vrho_up, exc_up, vrho_p, exc_p, cuda);

      // Compare spin-polarized to unpolarized results and check
      // for consistency
      for( int i=0;i<70; ++i){
        ASSERT_LT(abs(exc_up[i]-exc_p[i]),TOL);
      }

      for( int i=0;i<70; ++i){
        ASSERT_LT(abs(vrho_up[i]-vrho_p[i]),TOL);
      }
      for( int i=0;i<70; ++i){
        ASSERT_LT(abs(vrho_up[i]-vrho_p[i+70]),TOL);
      }
    }
}

TEST(grid, testhmgga){
    double vrho_up[100];
    double vsigma_up[100];
    double exc_up[100];
    double vrho_p[200];
    double vsigma_p[300];
    double exc_p[100];
    int cuda=0;
    // for (cuda=0;cuda<test_cuda()+1;++cuda){
    for (cuda=0;cuda< 1;++cuda){
      test_hm_gga(vrho_up, vsigma_up, exc_up, vrho_p, vsigma_p, exc_p, cuda);

      // Compare spin-polarized to unpolarized results and check
      // for consistency
      for( int i=0;i<100; ++i){
        ASSERT_LT(abs(exc_up[i]-exc_p[i]),TOL);
      }

      for( int i=0;i<100; ++i){
        ASSERT_LT(abs(vrho_up[i]-vrho_p[i]),TOL);
        ASSERT_LT(abs(vrho_up[i]-vrho_p[i+100]),TOL);
        ASSERT_LT(abs(vsigma_up[i]-vsigma_p[i]),TOL);
        ASSERT_LT(abs(vsigma_up[i]-vsigma_p[i+100]*0.5),TOL);
        ASSERT_LT(abs(vsigma_up[i]-vsigma_p[i+200]),TOL);
      }
    }
}
TEST(grid, testhmmetagga){
    double vrho_up[100];
    double vsigma_up[100];
    double vlapl_up[100];
    double vtau_up[100];
    double exc_up[100];
    double vrho_p[200];
    double vsigma_p[300];
    double exc_p[100];
    double vlapl_p[200];
    double vtau_p[200];

    int cuda=0;
    for (cuda=0;cuda<test_cuda()+1;++cuda){
      test_hm_mgga(vrho_up, vsigma_up, vlapl_up,vtau_up, exc_up, vrho_p,
        vsigma_p, vlapl_p, vtau_p, exc_p, cuda);

      // Compare spin-polarized to unpolarized results and check
      // for consistency
      for( int i=0;i<100; ++i){
        ASSERT_LT(abs(exc_up[i]-exc_p[i]),TOL);
      }

      for( int i=0;i<100; ++i){
        ASSERT_LT(abs(vrho_up[i]-vrho_p[i]),TOL);
        ASSERT_LT(abs(vrho_up[i]-vrho_p[i+100]),TOL);
        ASSERT_LT(abs(vsigma_up[i]-vsigma_p[i]),TOL);
        ASSERT_LT(abs(vsigma_up[i]-vsigma_p[i+100]*0.5),TOL);
        ASSERT_LT(abs(vsigma_up[i]-vsigma_p[i+200]),TOL);
        ASSERT_LT(abs(vtau_up[i]-vtau_p[i]),TOL);
        ASSERT_LT(abs(vtau_up[i]-vtau_p[i+100]),TOL);
        ASSERT_LT(abs(vlapl_up[i]),TOL);
        ASSERT_LT(abs(vlapl_p[i]),TOL);
        ASSERT_LT(abs(vlapl_p[i+100]),TOL);
      }
    }
}
}
GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from gtest_main.cc\n");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
