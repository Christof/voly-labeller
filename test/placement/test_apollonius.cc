#include "../test.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <Eigen/Core>
#include "../cuda_array_mapper.h"

void callApollonoius(std::vector<Eigen::Vector4f> &image);

TEST(Test_Apollonius, Apollonoius)
{
  std::vector<Eigen::Vector4f> image;
  image.push_back(Eigen::Vector4f::Zero());
  image.push_back(Eigen::Vector4f::Zero());
  image.push_back(Eigen::Vector4f::Zero());
  image.push_back(Eigen::Vector4f::Zero());

  image.push_back(Eigen::Vector4f::Zero());
  image.push_back(Eigen::Vector4f::Ones());
  image.push_back(Eigen::Vector4f::Zero());
  image.push_back(Eigen::Vector4f::Zero());

  image.push_back(Eigen::Vector4f::Zero());
  image.push_back(Eigen::Vector4f::Zero());
  image.push_back(Eigen::Vector4f::Zero());
  image.push_back(Eigen::Vector4f::Zero());

  image.push_back(Eigen::Vector4f::Zero());
  image.push_back(Eigen::Vector4f::Zero());
  image.push_back(Eigen::Vector4f::Zero());
  image.push_back(Eigen::Vector4f::Zero());

  callApollonoius(image);

  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[0], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[1], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[2], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[3], 1e-4f);

  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[4], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[5], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[6], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[7], 1e-4f);

  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[8], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[9], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[10], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[11], 1e-4f);

  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[12], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[13], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[14], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(1, 0, 0, 1), image[15], 1e-4f);
}

