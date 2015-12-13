#include "../test.h"
#include <QFile>
#include "../../src/placement/constraint_updater.h"
#include "../../src/graphics/qimage_drawer.h"

TEST(Test_ConstraintUpdater, Draw)
{
  int width = 512;
  int height = 256;
  auto drawer = std::make_shared<Graphics::QImageDrawer>(width, height);
  ConstraintUpdater constraintUpdater(drawer, width, height);

  constraintUpdater.clear();

  Eigen::Vector2i anchorPosition(100, 50);
  Eigen::Vector2i labelSize(62, 20);
  Eigen::Vector2i lastAnchorPosition(200, 70);
  Eigen::Vector2i lastLabelPosition(150, 120);
  Eigen::Vector2i lastLabelSize(60, 40);
  constraintUpdater.drawConstraintRegionFor(anchorPosition, labelSize,
                                            lastAnchorPosition,
                                            lastLabelPosition, lastLabelSize);

  drawer->image->save("constraints.png");

  QFile expectedFile("expected-constraints.png");
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

