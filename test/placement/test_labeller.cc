#include "../test.h"
#include "../../src/placement/labeller.h"
#include "../../src/placement/constraint_updater.h"
#include "../../src/labelling/labels.h"
#include "../../src/utils/image_persister.h"
#include "../../src/utils/path_helper.h"
#include "../../src/graphics/qimage_drawer.h"
#include "../cuda_array_mapper.h"

TEST(Test_PlacementLabeller, Integration)
{
  const int width = 512;
  const int height = 512;

  auto labels = std::make_shared<Labels>();
  labels->add(Label(1, "Shoulder", Eigen::Vector3f(0.174f, 0.55f, 0.034f)));
  labels->add(Label(2, "Ellbow", Eigen::Vector3f(0.34f, 0.322f, -0.007f)));
  labels->add(Label(3, "Wound", Eigen::Vector3f(0.262f, 0.422f, 0.058f),
                    Eigen::Vector2i(128, 128)));
  labels->add(Label(4, "Wound 2", Eigen::Vector3f(0.034f, 0.373f, 0.141f)));

  Placement::Labeller labeller(labels);

  cudaChannelFormatDesc floatChannelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  auto occupancyImage =
      ImagePersister::loadR32F(absolutePathOfProjectRelativePath(
          std::string("assets/tests/placement-labeller/occupancy.tiff")));
  auto occupancyTextureMapper = std::make_shared<CudaArrayMapper<float>>(
      width, height, occupancyImage, floatChannelDesc);

  auto distanceTransformImage =
      ImagePersister::loadR32F(absolutePathOfProjectRelativePath(std::string(
          "assets/tests/placement-labeller/distanceTransform.tiff")));
  auto distanceTransformTextureMapper =
      std::make_shared<CudaArrayMapper<float>>(
          width, height, distanceTransformImage, floatChannelDesc);

  cudaChannelFormatDesc vector4ChannelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  std::vector<Eigen::Vector4f> apolloniusImage(width * height,
                                               Eigen::Vector4f(0, 0, 0, 0));
  auto apolloniusTextureMapper =
      std::make_shared<CudaArrayMapper<Eigen::Vector4f>>(
          width, height, apolloniusImage, vector4ChannelDesc);

  std::vector<float> constraintImage(width * height, 0);
  auto constraintTextureMapper = std::make_shared<CudaArrayMapper<float>>(
      width, height, constraintImage, floatChannelDesc);
  auto drawer = std::make_shared<Graphics::QImageDrawer>(width, height);
  auto constraintUpdater =
      std::make_shared<ConstraintUpdater>(drawer, width, height);

  labeller.initialize(occupancyTextureMapper, distanceTransformTextureMapper,
                      apolloniusTextureMapper, constraintTextureMapper,
                      constraintUpdater);
  labeller.resize(1024, 1024);

  Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
  view(2, 3) = -1.0f;
  Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
  view(2, 2) = -2.53385f;
  view(2, 3) = -1.94522f;
  view(3, 2) = -1.0f;
  LabellerFrameData frameData(0.02f, projection, view);
  auto newPositions = labeller.update(frameData);

  ASSERT_EQ(labels->count(), newPositions.size());

  Eigen::Vector3f expectedPosition1(0.192445f, 0.686766f, 0.034f);
  EXPECT_Vector3f_NEAR(expectedPosition1, newPositions[1], 1e-5f);
  Eigen::Vector3f expectedPosition2(0.338289f, 0.129809f, -0.00699999f);
  EXPECT_Vector3f_NEAR(expectedPosition2, newPositions[2], 1e-5f);
  Eigen::Vector3f expectedPosition3(0.459961f, 0.445242f, 0.058f);
  EXPECT_Vector3f_NEAR(expectedPosition3, newPositions[3], 1e-5f);
  Eigen::Vector3f expectedPosition4(-0.184551f, 0.60734f, 0.141f);
  EXPECT_Vector3f_NEAR(expectedPosition4, newPositions[4], 1e-5f);
}

