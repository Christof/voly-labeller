#include "../test.h"
#include "../../src/placement/constraint_updater.h"
#include "./mock_anchor_constraint_drawer.h"
#include "./mock_shadow_constraint_drawer.h"

class Test_ConstraintUpdater : public ::testing::Test
{
 protected:
  virtual void SetUp()
  {
    int width = 512;
    int height = 256;
    anchorConstraintDrawer =
        std::make_shared<Mock_AnchorConstraintDrawer>(width, height);
    connectorShadowDrawer =
        std::make_shared<Mock_ShadowConstraintDrawer>(width, height);
    shadowConstraintDrawer =
        std::make_shared<Mock_ShadowConstraintDrawer>(width, height);
    constraintUpdater = std::make_shared<ConstraintUpdater>(
        width, height, anchorConstraintDrawer, connectorShadowDrawer,
        shadowConstraintDrawer);
  }

 public:
  std::shared_ptr<ConstraintUpdater> constraintUpdater;
  std::shared_ptr<Mock_AnchorConstraintDrawer> anchorConstraintDrawer;
  std::shared_ptr<Mock_ShadowConstraintDrawer> connectorShadowDrawer;
  std::shared_ptr<Mock_ShadowConstraintDrawer> shadowConstraintDrawer;

  void drawConstraintRegion()
  {
    constraintUpdater->clear();

    Eigen::Vector2i anchorPosition(100, 50);
    Eigen::Vector2i labelSize(62, 20);
    Eigen::Vector2i lastAnchorPosition(200, 70);
    Eigen::Vector2i lastLabelPosition(150, 120);
    Eigen::Vector2i lastLabelSize(60, 40);
    std::vector<Eigen::Vector2f> anchorPositions = { anchorPosition.cast<float>() };
    constraintUpdater->drawRegionsForAnchors(anchorPositions, labelSize);
    constraintUpdater->drawConstraintRegionFor(
        anchorPosition, labelSize, lastAnchorPosition, lastLabelPosition,
        lastLabelSize);
    constraintUpdater->finish();
  }
};

TEST_F(Test_ConstraintUpdater, DrawWithConnectorShadows)
{
  constraintUpdater->setIsConnectorShadowEnabled(true);

  drawConstraintRegion();

  auto anchors = anchorConstraintDrawer->anchors;
  EXPECT_EQ(2, anchors.size());
  EXPECT_FLOAT_EQ(100.0f, anchors[0]);
  EXPECT_FLOAT_EQ(50.0f, anchors[1]);

  auto connectorSources = connectorShadowDrawer->sources;
  EXPECT_EQ(2, connectorSources.size());
  EXPECT_FLOAT_EQ(100.0f, connectorSources[0]);
  EXPECT_FLOAT_EQ(50.0f, connectorSources[1]);

  auto connectorStarts = connectorShadowDrawer->starts;
  EXPECT_EQ(2, connectorStarts.size());
  EXPECT_FLOAT_EQ(200.0f, connectorStarts[0]);
  EXPECT_FLOAT_EQ(70.0f, connectorStarts[1]);

  auto connectorEnds = connectorShadowDrawer->ends;
  EXPECT_EQ(2, connectorEnds.size());
  EXPECT_FLOAT_EQ(150.0f, connectorEnds[0]);
  EXPECT_FLOAT_EQ(120.0f, connectorEnds[1]);

  auto labelShadowSources = shadowConstraintDrawer->sources;
  EXPECT_EQ(4, labelShadowSources.size());
  EXPECT_FLOAT_EQ(100.0f, labelShadowSources[0]);
  EXPECT_FLOAT_EQ(50.0f, labelShadowSources[1]);
  EXPECT_FLOAT_EQ(100.0f, labelShadowSources[2]);
  EXPECT_FLOAT_EQ(50.0f, labelShadowSources[3]);

  auto labelShadowStarts = shadowConstraintDrawer->starts;
  EXPECT_EQ(4, labelShadowStarts.size());
  EXPECT_FLOAT_EQ(120.0f, labelShadowStarts[0]);
  EXPECT_FLOAT_EQ(140.0f, labelShadowStarts[1]);
  EXPECT_FLOAT_EQ(120.0f, labelShadowStarts[2]);
  EXPECT_FLOAT_EQ(100.0f, labelShadowStarts[3]);

  auto labelShadowEnds = shadowConstraintDrawer->ends;
  EXPECT_EQ(4, labelShadowEnds.size());
  EXPECT_FLOAT_EQ(120.0f, labelShadowEnds[0]);
  EXPECT_FLOAT_EQ(100.0f, labelShadowEnds[1]);
  EXPECT_FLOAT_EQ(180.0f, labelShadowEnds[2]);
  EXPECT_FLOAT_EQ(100.0f, labelShadowEnds[3]);
}

TEST_F(Test_ConstraintUpdater, DrawWithoutConnectorShadows)
{
  constraintUpdater->setIsConnectorShadowEnabled(false);

  drawConstraintRegion();

  auto anchors = anchorConstraintDrawer->anchors;
  EXPECT_EQ(2, anchors.size());
  EXPECT_FLOAT_EQ(100.0f, anchors[0]);
  EXPECT_FLOAT_EQ(50.0f, anchors[1]);

  auto connectorSources = connectorShadowDrawer->sources;
  EXPECT_EQ(0, connectorSources.size());

  auto connectorStarts = connectorShadowDrawer->starts;
  EXPECT_EQ(0, connectorStarts.size());

  auto connectorEnds = connectorShadowDrawer->ends;
  EXPECT_EQ(0, connectorEnds.size());

  auto labelShadowSources = shadowConstraintDrawer->sources;
  EXPECT_EQ(4, labelShadowSources.size());
  EXPECT_FLOAT_EQ(100.0f, labelShadowSources[0]);
  EXPECT_FLOAT_EQ(50.0f, labelShadowSources[1]);
  EXPECT_FLOAT_EQ(100.0f, labelShadowSources[2]);
  EXPECT_FLOAT_EQ(50.0f, labelShadowSources[3]);

  auto labelShadowStarts = shadowConstraintDrawer->starts;
  EXPECT_EQ(4, labelShadowStarts.size());
  EXPECT_FLOAT_EQ(120.0f, labelShadowStarts[0]);
  EXPECT_FLOAT_EQ(140.0f, labelShadowStarts[1]);
  EXPECT_FLOAT_EQ(120.0f, labelShadowStarts[2]);
  EXPECT_FLOAT_EQ(100.0f, labelShadowStarts[3]);

  auto labelShadowEnds = shadowConstraintDrawer->ends;
  EXPECT_EQ(4, labelShadowEnds.size());
  EXPECT_FLOAT_EQ(120.0f, labelShadowEnds[0]);
  EXPECT_FLOAT_EQ(100.0f, labelShadowEnds[1]);
  EXPECT_FLOAT_EQ(180.0f, labelShadowEnds[2]);
  EXPECT_FLOAT_EQ(100.0f, labelShadowEnds[3]);
}

