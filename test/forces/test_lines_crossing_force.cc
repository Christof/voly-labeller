#include "../test.h"
#include "../../src/forces/lines_crossing_force.h"
#include "../../src/forces/label_state.h"

namespace Forces
{

TEST(Test_LinesCrossingForce, PushesLabesInPerpendicularDirectionIfLinesCross)
{
  LinesCrossingForce force;
  force.weight = 1.0f;

  Eigen::Vector2f size(0.1f, 0.1f);
  LabelState label(1, "Tested label", Eigen::Vector3f(1, 1, 0), size);
  label.labelPosition2D = Eigen::Vector2f(1, 1);
  label.anchorPosition2D = Eigen::Vector2f(0, 0);

  LabelState other(2, "Other label", Eigen::Vector3f(1, 1, 0), size);
  other.labelPosition2D = Eigen::Vector2f(0, 1);
  other.anchorPosition2D = Eigen::Vector2f(1, 0);

  LabellerFrameData frameData(1.0f, Eigen::Matrix4f::Identity(),
                              Eigen::Matrix4f::Identity());
  auto labels = std::vector<LabelState>{ label, other };
  auto result = force.calculateForce(label, labels, frameData);

  Eigen::Vector2f expected = Eigen::Vector2f(-1, 1).normalized();
  EXPECT_Vector2f_NEAR(expected, result, 0.0001f);
}
}
