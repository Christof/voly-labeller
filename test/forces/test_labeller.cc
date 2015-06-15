#include "../test.h"
#include "../../src/forces/labeller.h"
#include "../../src/labelling/labels.h"

class TestLabeller : public ::testing::Test
{
 protected:
  virtual void SetUp()
  {
    labels = std::make_shared<Labels>();
    labeller = std::make_shared<Forces::Labeller>(labels);
  } 
  // virtual void TearDown() {}

  std::shared_ptr<Labels> labels;
  std::shared_ptr<Forces::Labeller> labeller;
};

TEST_F(TestLabeller, AddedLabelIsUpdated)
{
  Label label(1, "Label text", Eigen::Vector3f(1, 2, 3));
  labels->add(label);

  double frameTime = 1.0;
  Forces::LabellerFrameData frameData(frameTime, Eigen::Matrix4f::Identity(),
                                      Eigen::Matrix4f::Identity());

  auto oldPosition = labeller->getLabels()[0].labelPosition;
  auto newPositions = labeller->update(frameData);

  EXPECT_EQ(1, newPositions.size());
  EXPECT_NE(oldPosition.x(), newPositions[0].x());

  EXPECT_Vector3f_NEAR(newPositions[label.id],
                       labeller->getLabels()[0].labelPosition, 1E-5);
}

