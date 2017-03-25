#include "../../src/utils/transfer_function_parser.h"
#include <gtest/gtest.h>
#include <QGradient>

TEST(Test_TransferFunctionParser, parse)
{
  TransferFunctionParser parser("../assets/transferfunctions/test.gra");
  QGradient *gradient = parser.parse();

  EXPECT_EQ(3, gradient->stops().size());

  QGradientStop gradient0(0.298828, QColor(0, 0, 0, 0));
  EXPECT_FLOAT_EQ(gradient0.first, gradient->stops()[0].first);
  EXPECT_EQ(gradient0.second, gradient->stops()[0].second);

  QGradientStop gradient1(0.311035, QColor(0, 100, 0, 76));
  EXPECT_FLOAT_EQ(gradient1.first, gradient->stops()[1].first);
  EXPECT_EQ(gradient1.second, gradient->stops()[1].second);

  QGradientStop gradient2(0.37207, QColor(30, 10, 40, 255));
  EXPECT_FLOAT_EQ(gradient2.first, gradient->stops()[2].first);
  EXPECT_EQ(gradient2.second, gradient->stops()[2].second);

  EXPECT_EQ(QGradient::Type::LinearGradient, gradient->type());
}
