#include "../test.h"
#include "../image_comparer.h"
#include <QFile>
#include "../../src/placement/constraint_updater.h"
#include "../../src/placement/anchor_constraint_drawer.h"
#include "../../src/placement/shadow_constraint_drawer.h"
#include "../../src/graphics/qimage_drawer.h"

class Mock_AnchorConstraintDrawer : public AnchorConstraintDrawer
{
 public:
  Mock_AnchorConstraintDrawer(int width, int height)
    : AnchorConstraintDrawer(width, height)
  {
  }

  virtual void update(const std::vector<float> &anchors)
  {
  }
  virtual void draw(float color, Eigen::Vector2f halfSize)
  {
  }
  virtual void clear()
  {
  }
};

class Mock_ShadowConstraintDrawer : public ShadowConstraintDrawer
{
 public:
  Mock_ShadowConstraintDrawer(int width, int height)
    : ShadowConstraintDrawer(width, height){};

  virtual void update(const std::vector<float> &anchors){};

  virtual void update(const std::vector<float> &sources,
                      const std::vector<float> &starts,
                      const std::vector<float> &ends){};

  virtual void clear(){};
};

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
    std::vector<Eigen::Vector2i> anchorPositions = { anchorPosition };
    constraintUpdater->drawRegionsForAnchors(anchorPositions, labelSize);
    constraintUpdater->drawConstraintRegionFor(
        anchorPosition, labelSize, lastAnchorPosition, lastLabelPosition,
        lastLabelSize);
  }
};

TEST_F(Test_ConstraintUpdater, DrawWithConnectorShadows)
{
  constraintUpdater->setIsConnectorShadowEnabled(true);

  drawConstraintRegion();

  /*
  compareImages("expected-constraints-with-connectors.png",
                "constraints-with-connectors.png", drawer->image.get());
                */
}

TEST_F(Test_ConstraintUpdater, DrawWithoutConnectorShadows)
{
  constraintUpdater->setIsConnectorShadowEnabled(false);

  drawConstraintRegion();

  /*
  compareImages("expected-constraints-without-connectors.png",
                "constraints-without-connectors.png", drawer->image.get());
                */
}

