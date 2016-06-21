#include "../test.h"
#include "../../src/forces/labeller.h"
#include "../../src/labelling/labels.h"
#include "../../src/labelling/label_positions.h"

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

  LabelPositions getDefaultPlacementResult()
  {
    std::map<int, Eigen::Vector3f> positions3d = { { 1, Eigen::Vector3f(
                                                            1, 2, 2.5f) } };
    std::map<int, Eigen::Vector3f> positionsNDC = {
      { 1, Eigen::Vector3f(0.5, 0.7, 0.5f) }
    };

    return LabelPositions(positionsNDC, positions3d);
  }
};

TEST_F(TestLabeller, AddedLabelIsUpdated)
{
  auto oldPosition = labeller->getLabels()[0].labelPosition;
  labeller->forces[0]->isEnabled = true;

  auto newPositions =
      labeller->update(getDefaultFrameData(), getDefaultPlacementResult());

  EXPECT_EQ(1, newPositions.size());
  EXPECT_NE(oldPosition.x(), newPositions.get3dFor(label.id).x());
}

TEST_F(TestLabeller, FromUpdateReturnedPositionsMatchGetPositions)
{
  auto newPositions =
      labeller->update(getDefaultFrameData(), getDefaultPlacementResult());

  EXPECT_Vector3f_NEAR(newPositions.get3dFor(label.id),
                       labeller->getLabels()[0].labelPosition, 1E-5);
}

TEST_F(TestLabeller, LabelHasDepthOfPlacementResult)
{
  float depth3d = 2.0f;
  float depthNDC = 0.7f;
  LabelPositions placementResult;
  placementResult.update(label.id, Eigen::Vector3f(0, 0, depthNDC),
                         Eigen::Vector3f(0, 0, depth3d));

  auto newPositions = labeller->update(getDefaultFrameData(), placementResult);

  EXPECT_NEAR(depth3d, newPositions.get3dFor(label.id).z(), 1E-5);
  EXPECT_NEAR(depthNDC, newPositions.getNDCFor(label.id).z(), 1E-5);
}

TEST_F(TestLabeller,
       LabelUpdateSetsTheGivenAnchorPositionAndReinitializesTheLabelPosition)
{
  auto newPositions =
      labeller->update(getDefaultFrameData(), getDefaultPlacementResult());

  label.anchorPosition = Eigen::Vector3f(-1, -2, -3);
  labels->update(label);

  EXPECT_NE(newPositions.get3dFor(label.id).x(),
            labeller->getLabels()[0].labelPosition.x());
  EXPECT_NE(newPositions.get3dFor(label.id).y(),
            labeller->getLabels()[0].labelPosition.y());
  EXPECT_NE(newPositions.get3dFor(label.id).z(),
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
  EXPECT_Vector3f_NEAR(newPositions.get3dFor(label.id),
                       newLabelState.labelPosition, 1E-5);
}

