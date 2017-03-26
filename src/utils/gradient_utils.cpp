#include "./gradient_utils.h"
#include <Eigen/Core>
#include <QDomDocument>
#include <QPainter>
#include <QFile>
#include <vector>
#include <stdexcept>
#include "./transfer_function_parser.h"

QImage GradientUtils::gradientToImage(const QGradient &gradient, QSize size)
{
  QImage image(size, QImage::Format_ARGB32);
  QPainter p(&image);
  p.setCompositionMode(QPainter::CompositionMode_Source);

  const qreal scaleFactor = 0.999999;
  p.scale(scaleFactor, scaleFactor);
  QGradient grad = gradient;
  grad.setCoordinateMode(QGradient::StretchToDeviceMode);
  p.fillRect(QRect(0, 0, size.width(), size.height()), grad);
  p.drawRect(QRect(0, 0, size.width() - 1, size.height() - 1));

  return image;
}

QImage GradientUtils::loadGradientAsImage(QString path, QSize size)
{
  TransferFunctionParser parser(path);
  QGradient gradient = parser.parse();
  return gradientToImage(gradient, size);
}

Eigen::Vector4f toEigen(QColor color)
{
  return Eigen::Vector4f(color.redF(), color.greenF(), color.blueF(),
                         color.alphaF());
}

void addColorTo(std::vector<float> &vector, Eigen::Vector4f color)
{
  vector.push_back(color.x());
  vector.push_back(color.y());
  vector.push_back(color.z());
  vector.push_back(color.w());
}

Eigen::Vector4f interpolateColors(Eigen::Vector4f first, Eigen::Vector4f second,
                                  float alpha)
{
  return (1.0f - alpha) * first + alpha * second;
}

std::vector<float>
GradientUtils::loadGradientAsFloats(const QGradient &gradient, int length,
                                    bool preMultiply)
{
  std::vector<float> result;
  auto stops = gradient.stops();
  Eigen::Vector4f beforeColor = toEigen(stops.first().second);
  float beforePoint = stops.first().first;
  int afterIndex = 1;

  if (beforePoint != 0.0f)
  {
    beforePoint = 0.0f;
    afterIndex = 0;
  }
  Eigen::Vector4f afterColor = toEigen(stops.at(afterIndex).second);
  float afterPoint = stops.at(afterIndex).first;
  float divisor = length - 1;

  for (int i = 0; i < length; ++i)
  {
    float progress = i / divisor;
    if (progress > afterPoint)
    {
      beforeColor = afterColor;
      beforePoint = afterPoint;

      ++afterIndex;
      if (afterIndex < stops.size())
      {
        auto newAfterStop = stops.at(afterIndex);
        afterColor = toEigen(newAfterStop.second);
        afterPoint = newAfterStop.first;
      }
      else
      {
        afterPoint = 1.0f;
      }
    }

    float alpha = (progress - beforePoint) / (afterPoint - beforePoint);
    Eigen::Vector4f color = interpolateColors(beforeColor, afterColor, alpha);
    if (preMultiply)
      color = Eigen::Vector4f(color.x() * color.w(), color.y() * color.w(),
                              color.z() * color.w(), color.w());

    addColorTo(result, color);
  }

  return result;
}

std::vector<float> GradientUtils::loadGradientAsFloats(QString path, int length,
                                                       bool preMultiply)
{
  TransferFunctionParser parser(path);
  QGradient gradient = parser.parse();
  return loadGradientAsFloats(gradient, length, preMultiply);
}

