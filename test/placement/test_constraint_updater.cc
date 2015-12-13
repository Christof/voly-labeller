#include "../test.h"
#include <QImage>
#include <QPainter>
#include <QFile>
#include "../../src/placement/constraint_updater.h"

class QImageDrawer : public Graphics::Drawer
{
 public:
  QImage *image;
  QImageDrawer(int width, int height)
  {
    image = new QImage(width, height, QImage::Format_Grayscale8);
  }

  void drawElementVector(std::vector<float> positions)
  {
    QPainter painter;
    painter.begin(image);
    painter.setBrush(QBrush(Qt::GlobalColor::white));
    painter.setPen(Qt::GlobalColor::white);

    std::vector<QPointF> points;
    for (size_t i = 0; i < positions.size() / 2; ++i)
    {
      float x = positions[i * 2];
      float y = positions[i * 2 + 1];
      if (i * 2 + 3 < positions.size() && x == positions[i * 2 + 2] &&
          y == positions[i * 2 + 3])
      {
        painter.drawConvexPolygon(points.data(), points.size());
        points.clear();
        i += 2;
      }
      points.push_back(QPointF(x, y));
    }
    painter.drawPolygon(points.data(), points.size(),
                        Qt::FillRule::OddEvenFill);
    painter.end();
  }

  void clear()
  {
    image->fill(Qt::GlobalColor::black);
  }
};

TEST(Test_ConstraintUpdater, Draw)
{
  int width = 512;
  int height = 256;
  auto drawer = std::make_shared<QImageDrawer>(width, height);
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

