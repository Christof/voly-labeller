#include "../test.h"
#include "../../src/forces/anchor_force.h"
#include "../../src/forces/label_state.h"

namespace Forces
{

TEST(Test_AnchorForce, PullsLabelToAnchor)
{
  AnchorForce force;
  force.weight = 1.0f;

  Eigen::Vector2f size(0.1f, 0.1f);
  LabelState label(1, "Tested label", Eigen::Vector3f(1, 1, 0), size);
  label.labelPosition2D = Eigen::Vector2f(1, 1);
  label.anchorPosition2D = Eigen::Vector2f(0, 0);

  LabellerFrameData frameData(1.0f, Eigen::Matrix4f::Identity(),
                              Eigen::Matrix4f::Identity());
  auto labels = std::vector<LabelState>{ label };
  auto result = force.calculateForce(label, labels, frameData);

  Eigen::Vector2f expected = Eigen::Vector2f(-1, -1) * 0.085786f;
  EXPECT_Vector2f_NEAR(expected, result, 0.0001f);
}

TEST(Test_Force, ReturnsZeroVectorIfForceIsDisabled)
{
  AnchorForce force;
  force.isEnabled = false;

  Eigen::Vector2f size(0.1f, 0.1f);
  LabelState label(1, "Tested label", Eigen::Vector3f(1, 1, 0), size);
  LabellerFrameData frameData(1.0f, Eigen::Matrix4f::Identity(),
                              Eigen::Matrix4f::Identity());
  auto labels = std::vector<LabelState>{ label };
  auto result = force.calculateForce(label, labels, frameData);

  EXPECT_Vector2f_NEAR(Eigen::Vector2f::Zero(), result, 1E-5f);
}

}
