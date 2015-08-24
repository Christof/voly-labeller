#include "../test.h"
#include "../../src/utils/gradient_utils.h"
#include <QGradient>

TEST(Test_GradientUtils, LoadFromFile)
{
  QGradient gradient =
      GradientUtils::loadGradient("../assets/transferfunctions/scapula1.gra");

  EXPECT_EQ(6, gradient.stops().size());
  EXPECT_EQ(QGradient::Type::LinearGradient, gradient.type());
}

TEST(Test_GradientUtils, LoadFromFileAsImage)
{
  QImage image = GradientUtils::loadGradientAsImage(
      "../assets/transferfunctions/MANIX_.gra", QSize(512, 10));

  // image.save("test.png");
  EXPECT_FALSE(image.isNull());
}

TEST(Test_GradientUtils, LoadGradientAsFloats)
{
  QGradient gradient;
  gradient.setColorAt(0, QColor(255, 127, 0, 10));
  gradient.setColorAt(1, QColor(0, 0, 255, 10));
  auto vector = GradientUtils::loadGradientAsFloats(gradient, 2);

  EXPECT_EQ(8, vector.size());
}
