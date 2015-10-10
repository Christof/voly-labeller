#include "../test.h"
#include <Eigen/Core>

void callApollonoius(std::vector<Eigen::Vector4f> &image);
std::vector<Eigen::Vector4f>
callDistanceTransform(std::vector<float> depth,
                      std::vector<float> &resultVector);

/*
TEST(Test_DistanceTransform, Apollonoius)
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

  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[0], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[1], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[2], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[3], 1e-4f);

  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[4], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[5], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[6], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[7], 1e-4f);

  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[8], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[9], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[10], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[11], 1e-4f);

  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[12], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[13], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[14], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[15], 1e-4f);
}
*/

TEST(Test_DistanceTransform, DistanceTransform)
{
  std::vector<float> depthImage;

  depthImage.push_back(1.0f);
  depthImage.push_back(1.0f);
  depthImage.push_back(1.0f);
  depthImage.push_back(1.0f);

  depthImage.push_back(1.0f);
  depthImage.push_back(0.0f);
  depthImage.push_back(0.0f);
  depthImage.push_back(1.0f);

  depthImage.push_back(1.0f);
  depthImage.push_back(0.0f);
  depthImage.push_back(0.0f);
  depthImage.push_back(1.0f);

  depthImage.push_back(1.0f);
  depthImage.push_back(1.0f);
  depthImage.push_back(1.0f);
  depthImage.push_back(1.0f);

  std::vector<float> resultVector;
  auto image = callDistanceTransform(depthImage, resultVector);

  EXPECT_EQ(16, resultVector.size());
  EXPECT_NEAR(0.0f, resultVector[0], 1e-4f);
  EXPECT_NEAR(0.0f, resultVector[1], 1e-4f);
  EXPECT_NEAR(0.0f, resultVector[2], 1e-4f);
  EXPECT_NEAR(0.0f, resultVector[3], 1e-4f);

  EXPECT_NEAR(0.0f, resultVector[4], 1e-4f);
  EXPECT_NEAR(1.0f, resultVector[5], 1e-4f);
  EXPECT_NEAR(1.0f, resultVector[6], 1e-4f);
  EXPECT_NEAR(1.0f, resultVector[7], 1e-4f);

  EXPECT_NEAR(0.0f, resultVector[8], 1e-4f);
  EXPECT_NEAR(1.0f, resultVector[9], 1e-4f);
  EXPECT_NEAR(2.0f, resultVector[10], 1e-4f);
  EXPECT_NEAR(2.0f, resultVector[11], 1e-4f);

  EXPECT_NEAR(0.0f, resultVector[12], 1e-4f);
  EXPECT_NEAR(1.0f, resultVector[13], 1e-4f);
  EXPECT_NEAR(2.0f, resultVector[14], 1e-4f);
  EXPECT_NEAR(3.0f, resultVector[15], 1e-4f);

  /*
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[0], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[1], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[2], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[3], 1e-4f);

  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[4], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[5], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[6], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[7], 1e-4f);

  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[8], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[9], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[10], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[11], 1e-4f);

  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[12], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[13], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[14], 1e-4f);
  EXPECT_Vector4f_NEAR(Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1), image[15], 1e-4f);
  */
}
