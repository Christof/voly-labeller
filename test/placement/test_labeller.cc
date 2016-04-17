#include "../test.h"
#include "../qimage_drawer_with_updating.h"
#include "../../src/placement/labeller.h"
#include "../../src/placement/constraint_updater.h"
#include "../../src/placement/persistent_constraint_updater.h"
#include "../../src/labelling/labels.h"
#include "../../src/utils/image_persister.h"
#include "../../src/utils/path_helper.h"

class Test_PlacementLabeller : public ::testing::Test
{
 protected:
  void createLabeller()
  {
    const int width = 512;
    const int height = 512;

    cudaChannelFormatDesc floatChannelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    auto integralCostsImage =
        ImagePersister::loadR32F(absolutePathOfProjectRelativePath(
            std::string("assets/tests/placement-labeller/integralCosts.tiff")));
    auto integralCostsTextureMapper = std::make_shared<CudaArrayMapper<float>>(
        width, height, integralCostsImage, floatChannelDesc);

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
    auto persistentConstraintUpdater =
        std::make_shared<PersistentConstraintUpdater>(constraintUpdater);

    labeller = std::make_shared<Placement::Labeller>(labels);
    labeller->initialize(integralCostsTextureMapper, distanceTransformTextureMapper,
                         apolloniusTextureMapper, constraintTextureMapper,
                         persistentConstraintUpdater);
    labeller->resize(1024, 1024);
  }

  virtual void SetUp()
  {
    labels = std::make_shared<Labels>();
    labels->add(Label(1, "Shoulder", Eigen::Vector3f(0.174f, 0.55f, 0.034f)));
    labels->add(Label(2, "Ellbow", Eigen::Vector3f(0.34f, 0.322f, -0.007f)));
    labels->add(Label(3, "Wound", Eigen::Vector3f(0.262f, 0.422f, 0.058f),
                      Eigen::Vector2i(128, 128)));
    labels->add(Label(4, "Wound 2", Eigen::Vector3f(0.034f, 0.373f, 0.141f)));

    createLabeller();
  }

 public:
  std::shared_ptr<Labels> labels;
  std::shared_ptr<Placement::Labeller> labeller;
};

TEST_F(Test_PlacementLabeller, UpdateCalculatesPositionsFromRealData)
{
  Placement::CostFunctionWeights weights;
  weights.labelShadowConstraint = 1e2f;
  weights.occupancy = 1.0f;
  weights.distanceToAnchor = 1e-3f;
  weights.favorHorizontalOrVerticalLines = 1e-1f;
  weights.connectorShadowConstraint = 1e1f;
  labeller->setCostFunctionWeights(weights);

  Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
  view(2, 3) = -1.0f;
  Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
  view(2, 2) = -2.53385f;
  view(2, 3) = -1.94522f;
  view(3, 2) = -1.0f;
  LabellerFrameData frameData(0.02f, projection, view);
  auto newPositions = labeller->update(frameData);
  auto lastPlacementResult = labeller->getLastPlacementResult();

  labeller->cleanup();

  ASSERT_EQ(labels->count(), newPositions.size());

  std::vector<Eigen::Vector3f> expectedPositions = {
    Eigen::Vector3f(0.192445f, 0.686766f, 0.034f),
    Eigen::Vector3f(0.338289f, 0.129809f, -0.00699999f),
    Eigen::Vector3f(0.459961f, 0.445242f, 0.058f),
    Eigen::Vector3f(0.379167f, 0.446277f, 0.141f),
  };

  for (size_t i = 0; i < newPositions.size(); ++i)
  {
    EXPECT_Vector3f_NEAR(expectedPositions[i], newPositions[i + 1], 1e-5f);
    EXPECT_Vector3f_NEAR(expectedPositions[i], lastPlacementResult[i + 1],
                         1e-5f);
  }
}

TEST(Test_PlacementLabellerWithoutFixture, UpdatesReturnsEmptyMapIfLabellerIsNotInitialized)
{
  auto labeller = std::make_shared<Placement::Labeller>(std::make_shared<Labels>());
  LabellerFrameData frameData(0.02f, Eigen::Matrix4f::Identity(),
                              Eigen::Matrix4f::Identity());

  auto newPositions = labeller->update(frameData);
  auto lastPlacementResult = labeller->getLastPlacementResult();

  EXPECT_EQ(0, newPositions.size());
  EXPECT_EQ(0, lastPlacementResult.size());
}

