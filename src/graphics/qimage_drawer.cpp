#include "./qimage_drawer.h"
#include <QPainter>
#include <vector>

namespace Graphics
{

QImageDrawer::QImageDrawer(int width, int height)
{
  image = std::make_shared<QImage>(width, height, QImage::Format_Grayscale8);
}

void drawPolygon(QPainter &painter, std::vector<QPointF> &points)
{
  if (points.size() < 3)
    return;

  QPointF triangle[3];
  triangle[0] = points[0];
  for (size_t i = 2; i < points.size(); ++i)
  {
    triangle[1] = points[i - 1];
    triangle[2] = points[i];
    painter.drawConvexPolygon(triangle, 3);
  }
  points.clear();
}

void QImageDrawer::drawElementVector(std::vector<float> positions, float color)
{
  QPainter painter;
  QImage tempImage(image->width(), image->height(), image->format());
  tempImage.fill(Qt::GlobalColor::black);
  painter.begin(&tempImage);
  QColor brushColor = QColor::fromRgbF(color, color, color, 1.0f);
  painter.setBrush(QBrush(brushColor));
  painter.setPen(brushColor);

  std::vector<QPointF> points;
  size_t i = 0;
  if (positions[0] == positions[2] && positions[1] == positions[3])
    i = 1;

  for (; i < positions.size() / 2; ++i)
  {
    float x = positions[i * 2];
    float y = image->height() - positions[i * 2 + 1];
    if (i * 2 + 3 < positions.size() && x == positions[i * 2 + 2] &&
        y == positions[i * 2 + 3])
    {
      drawPolygon(painter, points);
      points.clear();
      i += 2;
    }
    points.push_back(QPointF(x, y));
  }
  drawPolygon(painter, points);
  painter.end();

  for (int x = 0; x < image->width(); ++x)
  {
    for (int y = 0; y < image->height(); ++y)
    {
      QRgb imagePixel = image->pixel(x, y);
      QRgb tempPixel = tempImage.pixel(x, y);
      image->setPixel(x, y, imagePixel | tempPixel);
    }
  }
}

void QImageDrawer::clear()
{
  image->fill(Qt::GlobalColor::black);
}

}  // namespace Graphics
