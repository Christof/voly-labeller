#include "../test.h"
#include "../../src/forces/center_force.h"
#include "../../src/forces/label_state.h"

namespace Forces
{

TEST(Test_CenterForce, PushesLabelAwayFromAnchorCenter)
{
  CenterForce force;
  force.weight = 1.0f;

  Eigen::Vector2f size(0.1f, 0.1f);
  LabelState label(1, "Tested label", Eigen::Vector3f(2, 2, 0), size);
  label.anchorPosition2D = Eigen::Vector2f(1, 1);

  LabelState other(2, "Other label", Eigen::Vector3f(0, 0, 0), size);
  other.anchorPosition2D = Eigen::Vector2f(-1, 1);

  // center is at (0, 1)

  LabellerFrameData frameData(1.0f, Eigen::Matrix4f::Identity(),
                              Eigen::Matrix4f::Identity());
  auto labels = std::vector<LabelState>{ label, other };
  force.beforeAll(labels);
  auto result = force.calculateForce(label, labels, frameData);

  EXPECT_Vector2f_NEAR(Eigen::Vector2f(1, 0), result, 0.0001f);
}

}
