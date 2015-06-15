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

    labels->add(label);
  }

  std::shared_ptr<Labels> labels;
  std::shared_ptr<Forces::Labeller> labeller;

  Label label = { 1, "Label text", Eigen::Vector3f(1, 2, 3) };

  Forces::LabellerFrameData getDefaultFrameData()
  {
    double frameTime = 1.0;
    return Forces::LabellerFrameData(frameTime, Eigen::Matrix4f::Identity(),
                                     Eigen::Matrix4f::Identity());
  }
};

TEST_F(TestLabeller, AddedLabelIsUpdated)
{
  auto oldPosition = labeller->getLabels()[0].labelPosition;

  auto newPositions = labeller->update(getDefaultFrameData());

  EXPECT_EQ(1, newPositions.size());
  EXPECT_NE(oldPosition.x(), newPositions[label.id].x());
}

TEST_F(TestLabeller, FromUpdateReturnedPositionsMatchGetPositions)
{
  auto newPositions = labeller->update(getDefaultFrameData());

  EXPECT_Vector3f_NEAR(newPositions[label.id],
                       labeller->getLabels()[0].labelPosition, 1E-5);
}

TEST_F(TestLabeller, LabelHasSameDepthValueAsAnchor)
{
  auto newPositions = labeller->update(getDefaultFrameData());

  EXPECT_NEAR(label.anchorPosition.z(), newPositions[label.id].z(), 1E-5);
}

TEST_F(TestLabeller,
       LabelUpdateSetsTheGivenAnchorPositionAndReinitializesTheLabelPosition)
{
  auto newPositions = labeller->update(getDefaultFrameData());

  label.anchorPosition = Eigen::Vector3f(-1, -2, -3);
  labels->update(label);

  EXPECT_NE(newPositions[label.id].x(),
            labeller->getLabels()[0].labelPosition.x());
  EXPECT_NE(newPositions[label.id].y(),
            labeller->getLabels()[0].labelPosition.y());
  EXPECT_NE(newPositions[label.id].z(),
            labeller->getLabels()[0].labelPosition.z());
}

TEST_F(TestLabeller, LabelUpdateSetsTheCurrentLabelPositionIfAnchorsUnchanged)
{
  auto newPositions = labeller->update(getDefaultFrameData());

  label.text = "Some changed text";
  labels->update(label);

  auto newLabelState = labeller->getLabels()[0];
  EXPECT_EQ(label.text, newLabelState.text);
  EXPECT_Vector3f_NEAR(newPositions[label.id], newLabelState.labelPosition,
                       1E-5);
}

