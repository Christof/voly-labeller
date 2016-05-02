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
  if (std::isnan(label.placementPosition2D.x()) &&
      std::isnan(label.placementPosition2D.y()))
    return Eigen::Vector2f(0, 0);

  Eigen::Vector2f direction = label.placementPosition2D - label.labelPosition2D;

  if (direction.norm() < 0.05f)
    return direction;

  return direction.normalized();
}

}  // namespace Forces
