#ifndef SRC_UTILS_GRADIENT_UTILS_H_

#define SRC_UTILS_GRADIENT_UTILS_H_

#include <QGradient>
#include <QImage>
#include <vector>

/**
 * \brief Util functions to load gradient files
 *
 */
class GradientUtils
{
 public:
  static QImage gradientToImage(const QGradient &gradient, QSize size);
  static QImage loadGradientAsImage(QString path, QSize size);
  static std::vector<float> loadGradientAsFloats(const QGradient &gradient,
                                                 int length,
                                                 bool preMultiply = true);
  /**
   * \brief Loads a gradient file and converts it to a color array represented
   * by a float vector
   *
   * The resolution is determined by the \p length.
   *
   * Between the stops the color is interpolated in a linear fashion. If the
   * stops don't start at 0 or end at 1 the first or last stop's value is
   * used respectively.
   */
  static std::vector<float> loadGradientAsFloats(QString path, int length,
                                                 bool preMultiply = true);
};

#endif  // SRC_UTILS_GRADIENT_UTILS_H_
