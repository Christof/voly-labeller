#include "./label_collision_force.h"
#include <vector>
#include <QDebug>
#include "../math/aabb2d.h"
#include "./label_state.h"

namespace Forces
{
LabelCollisionForce::LabelCollisionForce() : Force(1.0f)
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

    Math::Aabb2d aabb(label.labelPosition2D, 0.5f * label.size);
    Math::Aabb2d otherAabb(otherLabel.labelPosition2D, 0.5f * otherLabel.size);

    if (aabb.testAabb(otherAabb))
    {
      qWarning() << "Collision between:" << label.text.c_str() << "and"
                 << otherLabel.text.c_str();
      Eigen::Vector2f diff = label.labelPosition2D - otherLabel.labelPosition2D;
      result += diff.normalized();
    }
  }

  return result;
}
}  // namespace Forces
