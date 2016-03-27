#include "../test.h"
#include <QFile>
#include "../../src/placement/constraint_updater.h"
#include "../../src/graphics/qimage_drawer.h"

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

  drawer->image->save("constraints-with-connectors.png");

  QFile expectedFile("expected-constraints-with-connectors.png");
  ASSERT_TRUE(expectedFile.exists())
      << "File 'expected-constraints.png' does not "
         "exists. Check 'constraints.png' and "
         "rename it if it is correct.";

  QImage expectedImage(expectedFile.fileName());
  ASSERT_EQ(expectedImage.width(), drawer->image->width());
  ASSERT_EQ(expectedImage.height(), drawer->image->height());

  for (int y = 0; y < expectedImage.width(); ++y)
    for (int x = 0; x < expectedImage.width(); ++x)
      EXPECT_EQ(expectedImage.pixel(x, y), drawer->image->pixel(x, y));
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

  drawer->image->save("constraints-without-connectors.png");

  QFile expectedFile("expected-constraints-without-connectors.png");
  ASSERT_TRUE(expectedFile.exists())
      << "File 'expected-constraints.png' does not "
         "exists. Check 'constraints.png' and "
         "rename it if it is correct.";

  QImage expectedImage(expectedFile.fileName());
  ASSERT_EQ(expectedImage.width(), drawer->image->width());
  ASSERT_EQ(expectedImage.height(), drawer->image->height());

  for (int y = 0; y < expectedImage.width(); ++y)
    for (int x = 0; x < expectedImage.width(); ++x)
      EXPECT_EQ(expectedImage.pixel(x, y), drawer->image->pixel(x, y));
}
