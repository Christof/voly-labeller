#include "../test.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include <Eigen/Core>
#include "../cuda_array_mapper.h"
#include "../../src/utils/image_persister.h"
#include "../../src/utils/path_helper.h"
#include "../../src/placement/apollonius.h"
#include "../../src/utils/cuda_helper.h"

std::shared_ptr<CudaArrayMapper<Eigen::Vector4f>>
createCudaArrayMapper(int width, int height, std::vector<Eigen::Vector4f> data)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  return std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(width, height, data,
                                                            channelDesc);
}

std::shared_ptr<CudaArrayMapper<float>>
createCudaArrayMapper(int width, int height, std::vector<float> data)
{
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  return std::make_shared<CudaArrayMapper<float>>(width, height, data,
                                                  channelDesc);
}

std::vector<int> callApollonoius(std::vector<Eigen::Vector4f> &image,
                                 std::vector<float> distances, int imageSize,
                                 std::vector<Eigen::Vector4f> labelsSeed)
{
  int labelCount = labelsSeed.size();
  auto imageMapper = createCudaArrayMapper(imageSize, imageSize, image);
  auto distancesMapper = createCudaArrayMapper(imageSize, imageSize, distances);

  Apollonius apollonius(distancesMapper, imageMapper, labelsSeed, labelCount);
  apollonius.run();

  image = imageMapper->copyDataFromGpu();

  imageMapper->unmap();

  std::vector<int> result;
  thrust::host_vector<int> labelIndices = apollonius.getIds();
  for (auto index : labelIndices)
    result.push_back(index);

  return result;
}

TEST(Test_Apollonius, Apollonius)
{
  std::vector<Eigen::Vector4f> image;
  image.push_back(Eigen::Vector4f::Zero());
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

  std::vector<float> distances(16, 0);
  distances[5] = 1.0f;
  distances[6] = 1.0f;
  distances[9] = 1.0f;
  distances[10] = 1.0f;

  std::vector<Eigen::Vector4f> labelsSeed = { Eigen::Vector4f(0, 2, 1, 1) };

  auto insertionOrder = callApollonoius(image, distances, 4, labelsSeed);

  ASSERT_EQ(1, insertionOrder.size());
  EXPECT_EQ(0, insertionOrder[0]);

  ASSERT_EQ(16, image.size());
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

TEST(Test_Apollonius, ApolloniusWithRealData)
{
  auto distances = ImagePersister::loadR32F(absolutePathOfProjectRelativePath(
      std::string("assets/tests/distanceTransform.tiff")));
  int imageSize = sqrt(distances.size());

  auto outputImage = std::vector<Eigen::Vector4f>(distances.size());

  std::vector<Eigen::Vector4f> labelsSeed = {
    Eigen::Vector4f(1, 302.122, 401.756, 1),
    Eigen::Vector4f(2, 342.435, 337.859, 1),
    Eigen::Vector4f(3, 327.202, 370.684, 1),
    Eigen::Vector4f(4, 266.133, 367.162, 1),
  };
  auto insertionOrder =
      callApollonoius(outputImage, distances, imageSize, labelsSeed);

  auto expected = ImagePersister::loadRGBA32F(absolutePathOfProjectRelativePath(
      std::string("assets/tests/apollonius.tiff")));

  ImagePersister::saveRGBA32F(outputImage.data(), imageSize, imageSize,
                              "ApolloniusWithRealDataOutput.tiff");

  int diffCount = 0;
  for (unsigned int i = 0; i < outputImage.size(); ++i)
  {
    if ((expected[i] - outputImage[i]).norm() > 1e-4f)
    {
      std::cout << "expected for index " << i << ": " << expected[i]
                << " but was: " << outputImage[i] << std::endl;
      diffCount++;
    }
  }

  EXPECT_LE(diffCount, 10);
}

TEST(Test_Apollonius, ApolloniusWithRealDataPeeling)
{
  auto distances = ImagePersister::loadR32F(absolutePathOfProjectRelativePath(
      std::string("assets/tests/distanceTransformPeeling.tiff")));
  int imageSize = sqrt(distances.size());

  auto outputImage = std::vector<Eigen::Vector4f>(distances.size());

  std::vector<Eigen::Vector4f> labelsSeed = {
    Eigen::Vector4f(1, 278.221, 352.606, 1),
    Eigen::Vector4f(2, 321.543, 223.339, 1),
    Eigen::Vector4f(3, 282.681, 267.453, 1),
    Eigen::Vector4f(4, 210.064, 311.382, 1),
  };

  auto imageMapper = createCudaArrayMapper(imageSize, imageSize, outputImage);
  auto distancesMapper = createCudaArrayMapper(imageSize, imageSize, distances);

  Apollonius apollonius(distancesMapper, imageMapper, labelsSeed,
                        labelsSeed.size());
  apollonius.run();
  outputImage = imageMapper->copyDataFromGpu();
  ImagePersister::saveRGBA32F(outputImage.data(), imageSize, imageSize,
                              "ApolloniusWithRealDataOutputPeeling.tiff");

  apollonius.extractUniqueBoundaryIndices();
  apollonius.updateInputCuda();

  size_t iterationCount = 0;
  size_t labelCount = labelsSeed.size();
  while (apollonius.vlk_map.size() < labelCount && iterationCount < labelCount)
  {
    apollonius.run();
    outputImage = imageMapper->copyDataFromGpu();
    ImagePersister::saveRGBA32F(outputImage.data(), imageSize, imageSize,
                                "ApolloniusWithRealDataOutputPeeling" +
                                    std::to_string(iterationCount) + ".tiff");
    apollonius.extractUniqueBoundaryIndices();
    apollonius.updateInputCuda();

    ++iterationCount;
  }

  outputImage = imageMapper->copyDataFromGpu();

  imageMapper->unmap();

  auto expected = ImagePersister::loadRGBA32F(absolutePathOfProjectRelativePath(
      std::string("assets/tests/apolloniusPeeling.tiff")));

  std::cout << "vlk_map" << std::endl;
  for(auto pair : apollonius.vlk_map)
  {
    std::cout << "\t image index: " << pair.first << " insertion index: " << pair.second << std::endl;
  }

  /*
  int diffCount = 0;
  for (unsigned int i = 0; i < outputImage.size(); ++i)
  {
    if ((expected[i] - outputImage[i]).norm() > 1e-4f)
    {
      std::cout << "expected for index " << i << ": " << expected[i]
                << " but was: " << outputImage[i] << std::endl;
      diffCount++;
    }
  }

  EXPECT_LE(diffCount, 10);
  */
}

