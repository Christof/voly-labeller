#include "./anchor_force.h"
#include <Eigen/Core>
#include "./label_state.h"

namespace Forces
{
Eigen::Vector3f AnchorForce::calculate(LabelState &label,
                                       std::vector<LabelState> &labels)
{
  return (label.anchorPosition - label.labelPosition).normalized() * 0.001f;
}
}  // namespace Forces
