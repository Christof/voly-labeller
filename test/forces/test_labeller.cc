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

  LabellerFrameData getDefaultFrameData()
  {
    double frameTime = 1.0;
    return LabellerFrameData(frameTime, Eigen::Matrix4f::Identity(),
                             Eigen::Matrix4f::Identity());
  }

  std::map<int, Eigen::Vector3f> getDefaultPlacementResult()
  {
    return std::map<int, Eigen::Vector3f>{ { 1, Eigen::Vector3f(1, 2, 2.5f) } };
  }
};

TEST_F(TestLabeller, AddedLabelIsUpdated)
{
  auto oldPosition = labeller->getLabels()[0].labelPosition;
  labeller->forces[0]->isEnabled = true;

  auto newPositions =
      labeller->update(getDefaultFrameData(), getDefaultPlacementResult());

  EXPECT_EQ(1, newPositions.size());
  EXPECT_NE(oldPosition.x(), newPositions[label.id].x());
}

TEST_F(TestLabeller, FromUpdateReturnedPositionsMatchGetPositions)
{
  auto newPositions =
      labeller->update(getDefaultFrameData(), getDefaultPlacementResult());

  EXPECT_Vector3f_NEAR(newPositions[label.id],
                       labeller->getLabels()[0].labelPosition, 1E-5);
}

TEST_F(TestLabeller, LabelHasDepthOfPlacementResult)
{
  float depth = 2.0f;
  std::map<int, Eigen::Vector3f> placementResult{
    { label.id, Eigen::Vector3f(0, 0, depth) }
  };
  auto newPositions = labeller->update(getDefaultFrameData(), placementResult);

  EXPECT_NEAR(depth, newPositions[label.id].z(), 1E-5);
}

TEST_F(TestLabeller,
       LabelUpdateSetsTheGivenAnchorPositionAndReinitializesTheLabelPosition)
{
  auto newPositions =
      labeller->update(getDefaultFrameData(), getDefaultPlacementResult());

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
  auto newPositions =
      labeller->update(getDefaultFrameData(), getDefaultPlacementResult());

  label.text = "Some changed text";
  labels->update(label);

  auto newLabelState = labeller->getLabels()[0];
  EXPECT_EQ(label.text, newLabelState.text);
  EXPECT_Vector3f_NEAR(newPositions[label.id], newLabelState.labelPosition,
                       1E-5);
}

