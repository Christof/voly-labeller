#include "./label_collision_force.h"
#include "./label_state.h"

namespace Forces
{
LabelCollisionForce::LabelCollisionForce() : Force(0.01f)
{
}

Eigen::Vector3f
LabelCollisionForce::calculate(LabelState &label,
                               std::vector<LabelState> &labels,
                               const LabellerFrameData &frameData)
{
  Eigen::Vector3f result(0, 0, 0);

  for (auto &otherLabel : labels)
  {
    if (otherLabel.id == label.id)
      continue;

    result +=
        (label.labelPosition - otherLabel.labelPosition).normalized() * weight;
  }

  return result;
}
}  // namespace Forces
