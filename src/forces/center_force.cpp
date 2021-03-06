#include "./center_force.h"
#include <vector>
#include "./label_state.h"

namespace Forces
{
CenterForce::CenterForce() : Force("Center", 0.1f)
{
}

void CenterForce::beforeAll(std::vector<LabelState> &labels)
{
  Eigen::Vector2f anchorPositionSum(0, 0);
  for (auto &labelState : labels)
    anchorPositionSum += labelState.anchorPosition2D;

  if (labels.size() > 0)
    averageCenter = anchorPositionSum / labels.size();
  else
    averageCenter = Eigen::Vector2f(0, 0);
}

Eigen::Vector2f CenterForce::calculate(LabelState &label,
                                       std::vector<LabelState> &labels)
{
  return (label.anchorPosition2D - averageCenter).normalized();
}

}  // namespace Forces
