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

TEST(Test_GradientUtils, LoadGradientAsFloats_ForTwoStopsAnd2Pixels)
{
  QGradient gradient;
  gradient.setColorAt(0, QColor(255, 127, 0, 10));
  gradient.setColorAt(1, QColor(0, 0, 255, 10));
  auto vector = GradientUtils::loadGradientAsFloats(gradient, 2);

  EXPECT_EQ(8, vector.size());

  const float delta = 1e-4f;
  EXPECT_NEAR(1.0f, vector[0], delta);
  EXPECT_NEAR(0.498f, vector[1], delta);
  EXPECT_NEAR(0.0f, vector[2], delta);
  EXPECT_NEAR(0.0392f, vector[3], delta);

  EXPECT_NEAR(0.0f, vector[4], delta);
  EXPECT_NEAR(0.0f, vector[5], delta);
  EXPECT_NEAR(1.0f, vector[6], delta);
  EXPECT_NEAR(0.0392f, vector[7], delta);
}

TEST(Test_GradientUtils, LoadGradientAsFloats_TwoStopsAnd5Pixels)
{
  QGradient gradient;
  gradient.setColorAt(0, QColor(255, 127, 0, 10));
  gradient.setColorAt(1, QColor(0, 0, 255, 10));
  auto vector = GradientUtils::loadGradientAsFloats(gradient, 5);

  EXPECT_EQ(20, vector.size());

  const float delta = 1e-4f;
  EXPECT_NEAR(1.0f, vector[0], delta);
  EXPECT_NEAR(0.498f, vector[1], delta);
  EXPECT_NEAR(0.0f, vector[2], delta);
  EXPECT_NEAR(0.0392f, vector[3], delta);

  EXPECT_NEAR(0.75f, vector[4], delta);
  EXPECT_NEAR(0.37353f, vector[5], delta);
  EXPECT_NEAR(0.25f, vector[6], delta);
  EXPECT_NEAR(0.0392f, vector[7], delta);

  EXPECT_NEAR(0.5f, vector[8], delta);
  EXPECT_NEAR(0.24901f, vector[9], delta);
  EXPECT_NEAR(0.5f, vector[10], delta);
  EXPECT_NEAR(0.0392f, vector[11], delta);

  EXPECT_NEAR(0.25f, vector[12], delta);
  EXPECT_NEAR(0.1245f, vector[13], delta);
  EXPECT_NEAR(0.75f, vector[14], delta);
  EXPECT_NEAR(0.0392f, vector[15], delta);

  EXPECT_NEAR(0.0f, vector[16], delta);
  EXPECT_NEAR(0.0f, vector[17], delta);
  EXPECT_NEAR(1.0f, vector[18], delta);
  EXPECT_NEAR(0.0392f, vector[19], delta);
}
