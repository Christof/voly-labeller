#include "./placement_force.h"
#include <vector>
#include "./label_state.h"

namespace Forces
{

PlacementForce::PlacementForce() : Force("Placement", 2.0f)
{
}

Eigen::Vector2f PlacementForce::calculate(LabelState &label,
                                          std::vector<LabelState> &labels)
{
  return (label.placementPosition2D - label.labelPosition2D).normalized();
}

}  // namespace Forces
