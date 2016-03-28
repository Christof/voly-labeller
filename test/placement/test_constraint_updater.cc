#include "../test.h"
#include "../image_comparer.h"
#include <QFile>
#include "../../src/placement/constraint_updater.h"
#include "../../src/graphics/qimage_drawer.h"

// TODO refactor test. Extract image comparison logic
//
TEST(Test_ConstraintUpdater, DrawWithConnectorShadows)
{
  int width = 512;
  int height = 256;
  auto drawer = std::make_shared<Graphics::QImageDrawer>(width, height);
  ConstraintUpdater constraintUpdater(drawer, width, height);
  constraintUpdater.setIsConnectorShadowEnabled(true);

  constraintUpdater.clear();

  Eigen::Vector2i anchorPosition(100, 50);
  Eigen::Vector2i labelSize(62, 20);
  Eigen::Vector2i lastAnchorPosition(200, 70);
  Eigen::Vector2i lastLabelPosition(150, 120);
  Eigen::Vector2i lastLabelSize(60, 40);
  constraintUpdater.drawConstraintRegionFor(anchorPosition, labelSize,
                                            lastAnchorPosition,
                                            lastLabelPosition, lastLabelSize);

  compareImages("expected-constraints-with-connectors.png",
                "constraints-with-connectors.png", drawer->image.get());
}

TEST(Test_ConstraintUpdater, DrawWithoutConnectorShadows)
{
  int width = 512;
  int height = 256;
  auto drawer = std::make_shared<Graphics::QImageDrawer>(width, height);
  ConstraintUpdater constraintUpdater(drawer, width, height);
  constraintUpdater.setIsConnectorShadowEnabled(false);

  constraintUpdater.clear();

  Eigen::Vector2i anchorPosition(100, 50);
  Eigen::Vector2i labelSize(62, 20);
  Eigen::Vector2i lastAnchorPosition(200, 70);
  Eigen::Vector2i lastLabelPosition(150, 120);
  Eigen::Vector2i lastLabelSize(60, 40);
  constraintUpdater.drawConstraintRegionFor(anchorPosition, labelSize,
                                            lastAnchorPosition,
                                            lastLabelPosition, lastLabelSize);

  compareImages("expected-constraints-without-connectors.png",
                "constraints-without-connectors.png", drawer->image.get());
}

