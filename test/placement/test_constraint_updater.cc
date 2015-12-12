#include "../test.h"
#include <QImage>
#include <QPainter>
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
      points.push_back(QPointF(positions[i * 2], positions[i * 2 + 1]));
    }
    painter.drawPolygon(points.data(), points.size(),
                        Qt::FillRule::WindingFill);
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
  auto drawer =
      std::make_shared<QImageDrawer>(width, height);
  ConstraintUpdater constraintUpdater(drawer, width, height);

  constraintUpdater.clear();

  Eigen::Vector2i anchorPosition(100, 50);
  Eigen::Vector2i labelSize(62, 20);
  Eigen::Vector2i lastAnchorPosition(200, 70);
  Eigen::Vector2i lastLabelPosition(200, 50);
  Eigen::Vector2i lastLabelSize(60, 40);
  constraintUpdater.drawConstraintRegionFor(anchorPosition, labelSize,
                                            lastAnchorPosition,
                                            lastLabelPosition, lastLabelSize);

  drawer->image->save("constraints.png");
}

