#include "../test.h"
#include "../image_comparer.h"
#include <QFile>
#include "../../src/placement/constraint_updater.h"
#include "../../src/graphics/qimage_drawer.h"

class Test_ConstraintUpdater : public ::testing::Test
{
 protected:
  virtual void SetUp()
  {
    int width = 512;
    int height = 256;
    drawer = std::make_shared<Graphics::QImageDrawer>(width, height);
    constraintUpdater =
        std::make_shared<ConstraintUpdater>(drawer, width, height);
  }

 public:
  std::shared_ptr<Graphics::QImageDrawer> drawer;
  std::shared_ptr<ConstraintUpdater> constraintUpdater;

  void drawConstraintRegion()
  {
    constraintUpdater->clear();

    Eigen::Vector2i anchorPosition(100, 50);
    Eigen::Vector2i labelSize(62, 20);
    Eigen::Vector2i lastAnchorPosition(200, 70);
    Eigen::Vector2i lastLabelPosition(150, 120);
    Eigen::Vector2i lastLabelSize(60, 40);
    constraintUpdater->drawAnchorRegion(anchorPosition, labelSize);
    constraintUpdater->drawConstraintRegionFor(
        anchorPosition, labelSize, lastAnchorPosition, lastLabelPosition,
        lastLabelSize);
  }
};

TEST_F(Test_ConstraintUpdater, DrawWithConnectorShadows)
{
  constraintUpdater->setIsConnectorShadowEnabled(true);

  drawConstraintRegion();

  compareImages("expected-constraints-with-connectors.png",
                "constraints-with-connectors.png", drawer->image.get());
}

TEST_F(Test_ConstraintUpdater, DrawWithoutConnectorShadows)
{
  constraintUpdater->setIsConnectorShadowEnabled(false);

  drawConstraintRegion();

  compareImages("expected-constraints-without-connectors.png",
                "constraints-without-connectors.png", drawer->image.get());
}

