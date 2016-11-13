#include "../test.h"
#include "../qimage_drawer_with_updating.h"
#include "../../src/placement/labeller.h"
#include "../../src/placement/constraint_updater.h"
#include "../../src/placement/anchor_constraint_drawer.h"
#include "../../src/placement/shadow_constraint_drawer.h"
#include "../../src/placement/constraint_updater.h"
#include "../../src/placement/persistent_constraint_updater.h"
#include "../../src/placement/insertion_order_labels_arranger.h"
#include "../../src/labelling/labels.h"
#include "../../src/utils/image_persister.h"
#include "../../src/utils/path_helper.h"
#include "./mock_anchor_constraint_drawer.h"
#include "./mock_shadow_constraint_drawer.h"

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

    cudaChannelFormatDesc byteChannelDesc =
        cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    std::vector<unsigned char> constraintImage(width * height, 0);
    auto constraintTextureMapper =
        std::make_shared<CudaArrayMapper<unsigned char>>(
            width, height, constraintImage, byteChannelDesc);

    anchorConstraintDrawer =
        std::make_shared<Mock_AnchorConstraintDrawer>(width, height);
    connectorShadowDrawer =
        std::make_shared<Mock_ShadowConstraintDrawer>(width, height);
    shadowConstraintDrawer =
        std::make_shared<Mock_ShadowConstraintDrawer>(width, height);
    float scaleFactor = 1.0f;
    auto constraintUpdater = std::make_shared<ConstraintUpdater>(
        width, height, anchorConstraintDrawer, connectorShadowDrawer,
        shadowConstraintDrawer, scaleFactor);

    auto persistentConstraintUpdater =
        std::make_shared<PersistentConstraintUpdater>(constraintUpdater);

    labeller = std::make_shared<Placement::Labeller>(labels);
    labeller->initialize(integralCostsTextureMapper, constraintTextureMapper,
                         persistentConstraintUpdater);
    labeller->setLabelsArranger(
        std::make_shared<Placement::InsertionOrderLabelsArranger>());
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

  LabelPositions getOldLabelPositions()
  {
    LabelPositions labelPositions;
    labelPositions.update(1, Eigen::Vector3f(0.1796875f, 0.640625f, 0.0f), Eigen::Vector3f(0.174f, 0.55f, 0.034f));
    labelPositions.update(2, Eigen::Vector3f(0.335937f, 0.20703125, 0.0f), Eigen::Vector3f(0.34f, 0.322f, -0.007f));
    labelPositions.update(3, Eigen::Vector3f(0.27734375, 0.69921875f, 0.0f), Eigen::Vector3f(0.262f, 0.422f, 0.058f));
    labelPositions.update(3, Eigen::Vector3f(0.1875f, 0.63671875f, 0.0f), Eigen::Vector3f(0.034f, 0.373f, 0.141f));

    return labelPositions;
  }

 public:
  std::shared_ptr<Labels> labels;
  std::shared_ptr<Placement::Labeller> labeller;

  std::shared_ptr<Mock_AnchorConstraintDrawer> anchorConstraintDrawer;
  std::shared_ptr<Mock_ShadowConstraintDrawer> connectorShadowDrawer;
  std::shared_ptr<Mock_ShadowConstraintDrawer> shadowConstraintDrawer;
};

TEST_F(Test_PlacementLabeller, UpdateCalculatesPositionsFromRealData)
{
  Placement::CostFunctionWeights weights;
  weights.labelShadowConstraint = 1e2f;
  weights.integralCosts = 1.0f;
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
  auto newPositions = labeller->update(frameData, false, getOldLabelPositions());
  auto lastPlacementResult = labeller->getLastPlacementResult();

  EXPECT_FLOAT_EQ(0.62223798, labeller->getLastSumOfCosts());

  labeller->cleanup();

  ASSERT_EQ(labels->count(), newPositions.size());

  std::vector<Eigen::Vector2f> expectedPositions = {
    Eigen::Vector2f(0.1796875f, 0.640625f),
    Eigen::Vector2f(0.335937f, 0.20703125),
    Eigen::Vector2f(0.27734375f, 0.69921875f),
    Eigen::Vector2f(0.1875f, 0.63671875f),
  };

  for (size_t i = 0; i < newPositions.size(); ++i)
  {
    EXPECT_Vector2f_NEAR(expectedPositions[i], newPositions[i + 1], 1e-5f);
    EXPECT_Vector2f_NEAR(expectedPositions[i], lastPlacementResult[i + 1],
                         1e-5f);
  }

  EXPECT_EQ(6, connectorShadowDrawer->sources.size());
  EXPECT_EQ(6, connectorShadowDrawer->starts.size());
  EXPECT_EQ(6, connectorShadowDrawer->ends.size());

  EXPECT_EQ(12, shadowConstraintDrawer->sources.size());
  EXPECT_EQ(12, shadowConstraintDrawer->starts.size());
  EXPECT_EQ(12, shadowConstraintDrawer->ends.size());
}

TEST(Test_PlacementLabellerWithoutFixture,
     UpdatesReturnsEmptyMapIfLabellerIsNotInitialized)
{
  auto labeller =
      std::make_shared<Placement::Labeller>(std::make_shared<Labels>());
  LabellerFrameData frameData(0.02f, Eigen::Matrix4f::Identity(),
                              Eigen::Matrix4f::Identity());

  LabelPositions oldLabelPositions;
  auto newPositions = labeller->update(frameData, false, oldLabelPositions);
  auto lastPlacementResult = labeller->getLastPlacementResult();

  EXPECT_EQ(0, newPositions.size());
  EXPECT_EQ(0, lastPlacementResult.size());
}

