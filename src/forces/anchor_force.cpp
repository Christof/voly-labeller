#include "./anchor_force.h"
#include <Eigen/Core>
#include "./label_state.h"

namespace Forces
{
AnchorForce::AnchorForce() : Force(0.01f)
{
}

Eigen::Vector2f AnchorForce::calculate(LabelState &label,
                                       std::vector<LabelState> &labels,
                                       const LabellerFrameData &frameData)
{
  return (label.anchorPosition2D - label.labelPosition2D).normalized() * weight;
}
}  // namespace Forces
