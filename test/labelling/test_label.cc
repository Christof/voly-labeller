#include "../test.h"
#include "../../src/labelling/label.h"

TEST(Test_Label, EqualityIsDeterminedByIdTextAnchorPositionAndSize)
{
  Label label1(0, "Label 0", Eigen::Vector3f(1, 2, 3));
  Label label2(0, "Label 0", Eigen::Vector3f(1, 2, 3));

  EXPECT_TRUE(label1 == label2);
  EXPECT_FALSE(label1 != label2);
}

