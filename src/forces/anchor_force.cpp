#include "./anchor_force.h"
#include <Eigen/Core>
#include <vector>
#include <algorithm>
#include "./label_state.h"

namespace Forces
{
AnchorForce::AnchorForce() : Force(1.0f)
{
}

Eigen::Vector2f AnchorForce::calculate(LabelState &label,
                                       std::vector<LabelState> &labels,
                                       const LabellerFrameData &frameData)
{
  const float epsilon = 0.01f;
  Eigen::Vector2f diff = label.anchorPosition2D - label.labelPosition2D;
  float distance = diff.norm();
  float d = std::max(epsilon, distance);
  float factor = d > 0.2f ? 1.0f : -1.0f;

  return (d - 2 + 1 / d) * factor * diff.normalized();
}
}  // namespace Forces
