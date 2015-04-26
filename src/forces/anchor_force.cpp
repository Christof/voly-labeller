#include "./anchor_force.h"
#include <Eigen/Core>
#include "./label_state.h"

namespace Forces
{
AnchorForce::AnchorForce() : Force(0.01f)
{
}

Eigen::Vector3f AnchorForce::calculate(LabelState &label,
                                       std::vector<LabelState> &labels)
{
  return (label.anchorPosition - label.labelPosition).normalized() * weight;
}
}  // namespace Forces
