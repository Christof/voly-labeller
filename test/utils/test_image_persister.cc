#include "../test.h"
#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include "../../src/utils/image_persister.h"

TEST(Test_ImagePersister, LoadAndSave_RGBA32F) {
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

TEST(Test_ImagePersister, LoadAndSave_R32F)
{
  std::vector<float> image;
  image.push_back(0);
  image.push_back(0.1f);
  image.push_back(0.2f);
  image.push_back(0.2f);

  image.push_back(0.9f);
  image.push_back(0.1f);
  image.push_back(0.2f);
  image.push_back(0.2f);

  image.push_back(0.9f);
  image.push_back(0.1f);
  image.push_back(0.2f);
  image.push_back(0.2f);

  image.push_back(0.8f);
  image.push_back(0.1f);
  image.push_back(0.4f);
  image.push_back(0.2f);

  std::string filename = "Test_ImagePersisterR32F.tiff";
  ImagePersister::saveR32F(image.data(), 4, 4, filename);

  std::vector<float> loaded = ImagePersister::loadR32F(filename);

  ASSERT_EQ(image.size(), loaded.size());
  for (size_t i = 0; i < image.size(); ++i)
  {
    EXPECT_NEAR(image[i], loaded[i], 1e-2f);
  }

  boost::filesystem::remove(filename);
}

TEST(Test_ImagePersister, LoadAndSave_R8I)
{
  std::vector<unsigned char> image;
  image.push_back(0);
  image.push_back(127);
  image.push_back(32);
  image.push_back(255);

  image.push_back(123);
  image.push_back(132);
  image.push_back(221);
  image.push_back(10);

  image.push_back(11);
  image.push_back(14);
  image.push_back(70);
  image.push_back(83);

  image.push_back(84);
  image.push_back(98);
  image.push_back(100);
  image.push_back(71);

  std::string filename = "Test_ImagePersisterRI8.png";
  ImagePersister::saveR8I(image.data(), 4, 4, filename);

  std::vector<unsigned char> loaded = ImagePersister::loadR8I(filename);

  ASSERT_EQ(image.size(), loaded.size());
  for (size_t i = 0; i < image.size(); ++i)
  {
    EXPECT_EQ(image[i], loaded[i]);
  }

  boost::filesystem::remove(filename);
}
