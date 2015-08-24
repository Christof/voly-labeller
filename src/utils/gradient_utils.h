#ifndef SRC_UTILS_GRADIENT_UTILS_H_

#define SRC_UTILS_GRADIENT_UTILS_H_

#include <QGradient>
#include <QImage>
#include <vector>

/**
 * \brief
 *
 *
 */
class GradientUtils
{
 public:
  static QGradient loadGradient(QString path);
  static QImage gradientToImage(const QGradient &gradient, QSize size);
  static QImage loadGradientAsImage(QString path, QSize size);
  static std::vector<float> loadGradientAsFloats(QString path, int length);
};

#endif  // SRC_UTILS_GRADIENT_UTILS_H_
