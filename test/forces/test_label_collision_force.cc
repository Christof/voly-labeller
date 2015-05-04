#include "../test.h"
#include "../../src/forces/label_collision_force.h"
#include "../../src/forces/label_state.h"

namespace Forces
{

TEST(Test_LabelCollisionForce, NoForceIfLabelsDontCollide)
{
  LabelCollisionForce force;

  Eigen::Vector2f size(0.1f, 0.1f);
  LabelState label(1, "Tested label", Eigen::Vector3f(0, 0, 0), size);
  label.labelPosition2D = Eigen::Vector2f(0, 0);

  LabelState other(2, "Other label", Eigen::Vector3f(1, 1, 0), size);
  other.labelPosition2D = Eigen::Vector2f(1, 1);

  LabellerFrameData frameData(1.0f, Eigen::Matrix4f::Identity(),
                              Eigen::Matrix4f::Identity());
  auto labels = std::vector<LabelState>{ label, other };
  auto result = force.calculateForce(label, labels, frameData);

  EXPECT_Vector2f_NEAR(Eigen::Vector2f(0, 0), result, 0.0001f);
}

TEST(Test_LabelCollisionForce, ForceIfLabelsCollide)
{
  LabelCollisionForce force;

  Eigen::Vector2f size(1.0f, 1.0f);
  LabelState label(1, "Tested label", Eigen::Vector3f(0, 0, 0), size);
  label.labelPosition2D = Eigen::Vector2f(0, 0);

  LabelState other(2, "Other label", Eigen::Vector3f(1, 1, 0), size);
  other.labelPosition2D = Eigen::Vector2f(1, 1);

  LabellerFrameData frameData(1.0f, Eigen::Matrix4f::Identity(),
                              Eigen::Matrix4f::Identity());
  auto labels = std::vector<LabelState>{ label, other };
  auto result = force.calculateForce(label, labels, frameData);

  EXPECT_Vector2f_NEAR(Eigen::Vector2f(-0.70710678, -0.707010678), result, 0.0001f);
}

}  // namespace Forces
