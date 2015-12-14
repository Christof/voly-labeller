#include "./qimage_drawer.h"
#include <QPainter>
#include <vector>

namespace Graphics
{

QImageDrawer::QImageDrawer(int width, int height)
{
  image = std::make_shared<QImage>(width, height, QImage::Format_Grayscale8);
}

void QImageDrawer::drawElementVector(std::vector<float> positions)
{
  QPainter painter;
  painter.begin(image.get());
  painter.setBrush(QBrush(Qt::GlobalColor::white));
  painter.setPen(Qt::GlobalColor::white);

  std::vector<QPointF> points;
  size_t i = 0;
  if (positions[0] == positions[2] && positions[1] == positions[3])
    i = 1;

  for (; i < positions.size() / 2; ++i)
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
  painter.drawConvexPolygon(points.data(), points.size());
  painter.end();
}

void QImageDrawer::clear()
{
  image->fill(Qt::GlobalColor::black);
}

}  // namespace Graphics
