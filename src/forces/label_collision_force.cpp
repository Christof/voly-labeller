#include "./label_collision_force.h"
#include <vector>
#include "./label_state.h"

namespace Forces
{
LabelCollisionForce::LabelCollisionForce() : Force(0.01f)
{
}

Eigen::Vector2f LabelCollisionForce::calculate(LabelState &label,
                                               std::vector<LabelState> &labels)
{
  Eigen::Vector2f result(0, 0);

  for (auto &otherLabel : labels)
  {
    if (otherLabel.id == label.id)
      continue;

    Eigen::Vector2f diff = label.labelPosition2D - otherLabel.labelPosition2D;
    float factor = diff.norm() > 0.2f ? 0.0f : 1.0f;
    result += diff.normalized() * factor;
  }

  return result;
}
}  // namespace Forces
