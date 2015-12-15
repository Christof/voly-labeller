#include "../test.h"
#include "../../src/placement/labeller.h"
#include "../../src/placement/constraint_updater.h"
#include "../../src/labelling/labels.h"
#include "../../src/utils/image_persister.h"
#include "../../src/utils/path_helper.h"
#include "../../src/graphics/qimage_drawer.h"
#include "../cuda_array_mapper.h"

class QImageDrawerWithUpdating : public Graphics::QImageDrawer
{
 public:
  QImageDrawerWithUpdating(
      int width, int height,
      std::shared_ptr<CudaArrayMapper<unsigned char>> texture)
    : QImageDrawer(width, height), texture(texture)
  {
  }

  virtual void drawElementVector(std::vector<float> positions)
  {
    Graphics::QImageDrawer::drawElementVector(positions);

    const unsigned char *data = image->constBits();
    std::vector<unsigned char> newData(image->width() * image->height(), 0.0f);
    for (int i = 0; i < image->byteCount(); ++i)
    {
      newData[i] = data[i] > 0 || texture->getDataAt(i) > 0 ? 255 : 0;
    }

    texture->updateData(newData);
  }

  virtual void clear()
  {
    Graphics::QImageDrawer::clear();

    std::vector<unsigned char> newData(image->width() * image->height(), 0);
    texture->updateData(newData);
  }

 private:
  std::shared_ptr<CudaArrayMapper<unsigned char>> texture;
};

std::shared_ptr<Labels> createLabels()
{
  auto labels = std::make_shared<Labels>();
  labels->add(Label(1, "Shoulder", Eigen::Vector3f(0.174f, 0.55f, 0.034f)));
  labels->add(Label(2, "Ellbow", Eigen::Vector3f(0.34f, 0.322f, -0.007f)));
  labels->add(Label(3, "Wound", Eigen::Vector3f(0.262f, 0.422f, 0.058f),
                    Eigen::Vector2i(128, 128)));
  labels->add(Label(4, "Wound 2", Eigen::Vector3f(0.034f, 0.373f, 0.141f)));

  return labels;
}

std::shared_ptr<Placement::Labeller>
createLabeller(std::shared_ptr<Labels> labels)
{
  const int width = 512;
  const int height = 512;

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

  cudaChannelFormatDesc byteChannelDesc =
      cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
  std::vector<unsigned char> constraintImage(width * height, 0);
  auto constraintTextureMapper =
      std::make_shared<CudaArrayMapper<unsigned char>>(
          width, height, constraintImage, byteChannelDesc);
  auto drawer = std::make_shared<QImageDrawerWithUpdating>(
      width, height, constraintTextureMapper);
  auto constraintUpdater =
      std::make_shared<ConstraintUpdater>(drawer, width, height);

  auto labeller = std::make_shared<Placement::Labeller>(labels);
  labeller->initialize(occupancyTextureMapper, distanceTransformTextureMapper,
                       apolloniusTextureMapper, constraintTextureMapper,
                       constraintUpdater);
  labeller->resize(1024, 1024);

  return labeller;
}

TEST(Test_PlacementLabeller, Integration)
{

  auto labels = createLabels();
  auto labeller = createLabeller(labels);

  Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
  view(2, 3) = -1.0f;
  Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
  view(2, 2) = -2.53385f;
  view(2, 3) = -1.94522f;
  view(3, 2) = -1.0f;
  LabellerFrameData frameData(0.02f, projection, view);
  auto newPositions = labeller->update(frameData);

  ASSERT_EQ(labels->count(), newPositions.size());

  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.192445f, 0.686766f, 0.034f),
                       newPositions[1], 1e-5f);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.338289f, 0.129809f, -0.00699999f),
                       newPositions[2], 1e-5f);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(0.459961f, 0.445242f, 0.058f),
                       newPositions[3], 1e-5f);
  EXPECT_Vector3f_NEAR(Eigen::Vector3f(-0.184551f, 0.60734f, 0.141f),
                       newPositions[4], 1e-5f);
}

