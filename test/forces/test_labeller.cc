#include "../test.h"
#include "../../src/forces/labeller.h"
#include "../../src/labelling/labels.h"

TEST(Test_Labeller, AddedLabelIsUpdated)
{
  auto labels = std::make_shared<Labels>();
  Forces::Labeller labeller(labels);

  Label label(1, "Label text", Eigen::Vector3f(1, 2, 3));
  labels->add(label);

  double frameTime = 1.0;
  Forces::LabellerFrameData frameData(frameTime, Eigen::Matrix4f::Identity(),
                                      Eigen::Matrix4f::Identity());
  auto newPositions = labeller.update(frameData);

  EXPECT_EQ(1, newPositions.size());
  EXPECT_Vector3f_NEAR(newPositions[label.id],
                       labeller.getLabels()[0].labelPosition, 1E-5);
}
