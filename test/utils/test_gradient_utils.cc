#include "../test.h"
#include "../../src/utils/gradient_utils.h"

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
  auto vector = GradientUtils::loadGradientAsFloats(
      "../assets/transferfunctions/lung_vessels.gra", 2);

  EXPECT_EQ(8, vector.size());
}
