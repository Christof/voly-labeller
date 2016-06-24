#include "../test.h"
#include "../../src/forces/placement_force.h"
#include "../../src/forces/label_state.h"

namespace Forces
{

TEST(Test_PlacementForce, PullsLabelToPlacementResult)
{
  PlacementForce force;
  force.weight = 1.0f;

  Eigen::Vector2f size(0.1f, 0.1f);
  LabelState label(1, "Tested label", Eigen::Vector3f(1, 1, 0), size);
  label.labelPosition2D = Eigen::Vector2f(1, 1);
  label.placementPosition2D = Eigen::Vector2f(0, 0);

  LabellerFrameData frameData(1.0f, Eigen::Matrix4f::Identity(),
                              Eigen::Matrix4f::Identity());
  auto labels = std::vector<LabelState>{ label };
  auto result = force.calculateForce(label, labels, frameData);

  Eigen::Vector2f expected = Eigen::Vector2f(-1, -1).normalized();
  EXPECT_Vector2f_NEAR(expected, result, 0.0001f);
}

TEST(Test_PlacementForce, ScalesForceDownIfLabelIsAlreadyNearThePlacementResult)
{
  PlacementForce force;
  force.weight = 1.0f;

  Eigen::Vector2f size(0.1f, 0.1f);
  LabelState label(1, "Tested label", Eigen::Vector3f(1, 1, 0), size);
  label.labelPosition2D = Eigen::Vector2f(0.01f, 0.01f);
  label.placementPosition2D = Eigen::Vector2f(0, 0);

  LabellerFrameData frameData(1.0f, Eigen::Matrix4f::Identity(),
                              Eigen::Matrix4f::Identity());
  auto labels = std::vector<LabelState>{ label };
  auto result = force.calculateForce(label, labels, frameData);

  Eigen::Vector2f expected = Eigen::Vector2f(-0.707106f, -0.707106f);
  EXPECT_Vector2f_NEAR(expected, result, 0.0001f);
}

TEST(Test_PlacementForce, DoesntExertForceIfNoPlacementResultAvailable)
{
  PlacementForce force;
  force.weight = 1.0f;

  Eigen::Vector2f size(0.1f, 0.1f);
  LabelState label(1, "Tested label", Eigen::Vector3f(1, 1, 0), size);
  label.labelPosition2D = Eigen::Vector2f(1, 1);
  std::cout << label.placementPosition2D << std::endl;

  LabellerFrameData frameData(1.0f, Eigen::Matrix4f::Identity(),
                              Eigen::Matrix4f::Identity());
  auto labels = std::vector<LabelState>{ label };
  auto result = force.calculateForce(label, labels, frameData);

  EXPECT_Vector2f_NEAR(Eigen::Vector2f(0, 0), result, 0.0001f);
}

}
