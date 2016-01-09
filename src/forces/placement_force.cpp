#include "./placement_force.h"
#include <vector>
#include "./label_state.h"

namespace Forces
{

PlacementForce::PlacementForce() : Force("Placement", 4.0f)
{
}

Eigen::Vector2f PlacementForce::calculate(LabelState &label,
                                          std::vector<LabelState> &labels)
{
  if (isnan(label.placementPosition2D.x()) &&
      isnan(label.placementPosition2D.y()))
    return Eigen::Vector2f(0, 0);

  return (label.placementPosition2D - label.labelPosition2D).normalized();
}

}  // namespace Forces
