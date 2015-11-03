#include "../test.h"
#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include "../../src/utils/image_persister.h"

TEST(Test_ImagePersister, LoadAndSave)
{
  std::vector<Eigen::Vector4f> image;
  image.push_back(Eigen::Vector4f(0, 0.5f, 0.7f, 0.9f));
  image.push_back(Eigen::Vector4f(0.1f, 0.6f, 0.6f, 0.8f));
  image.push_back(Eigen::Vector4f(0.2f, 0.3f, 0.6f, 0.9f));
  image.push_back(Eigen::Vector4f(0.2f, 0.4f, 0.4f, 0.8f));

  image.push_back(Eigen::Vector4f(0, 0.2f, 0.4f, 0.7f));
  image.push_back(Eigen::Vector4f(0.1f, 0.9f, 0.2f, 0.6f));
  image.push_back(Eigen::Vector4f(0.2f, 0.2f, 0.2f, 0.8f));
  image.push_back(Eigen::Vector4f(0.2f, 0.1f, 0.3f, 0.9f));

  image.push_back(Eigen::Vector4f(0, 0.5f, 0.7f, 0.7f));
  image.push_back(Eigen::Vector4f(0.1f, 0.6f, 0.6f, 0.6f));
  image.push_back(Eigen::Vector4f(0.2f, 0.3f, 0.6f, 0.9f));
  image.push_back(Eigen::Vector4f(0.2f, 0.4f, 0.4f, 0.9f));

  image.push_back(Eigen::Vector4f(0, 0.5f, 0.7f, 0.9f));
  image.push_back(Eigen::Vector4f(0.1f, 0.6f, 0.6f, 0.9f));
  image.push_back(Eigen::Vector4f(0.4f, 0.8f, 0.4f, 0.8f));
  image.push_back(Eigen::Vector4f(0.2f, 0.7f, 0.4f, 1.0f));

  std::string filename = "Test_ImagePersister.tiff";
  ImagePersister::saveRGBA32F(image.data(), 4, 4, filename);

  std::vector<Eigen::Vector4f> loaded = ImagePersister::loadRGBA32F(filename);

  ASSERT_EQ(image.size(), loaded.size());
  for (size_t i = 0; i < image.size(); ++i)
  {
    EXPECT_Vector4f_NEAR(image[i], loaded[i], 1e-2f);
  }

  boost::filesystem::remove(filename);
}
