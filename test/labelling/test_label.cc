#include "../test.h"
#include "../../src/labelling/label.h"

TEST(Test_Label, EqualityIsDeterminedByIdTextAnchorPositionAndSize)
{
  Label label1(0, "Label 0", Eigen::Vector3f(1, 2, 3));
  Label label2(0, "Label 0", Eigen::Vector3f(1, 2, 3));
  Label label3(1, "Label 0", Eigen::Vector3f(1, 2, 3));
  Label label4(0, "Label 1", Eigen::Vector3f(1, 2, 3));
  Label label5(0, "Label 0", Eigen::Vector3f(0, 0, 0));

  EXPECT_TRUE(label1 == label2);
  EXPECT_FALSE(label1 != label2);

  EXPECT_FALSE(label1 == label3);
  EXPECT_TRUE(label1 != label3);

  EXPECT_FALSE(label2 == label3);
  EXPECT_TRUE(label2 != label3);

  EXPECT_FALSE(label1 == label4);
  EXPECT_TRUE(label1 != label4);

  EXPECT_FALSE(label2 == label4);
  EXPECT_TRUE(label2 != label4);

  EXPECT_FALSE(label1 == label5);
  EXPECT_TRUE(label1 != label5);
}
