#include "./center_force.h"
#include <vector>
#include "./label_state.h"

namespace Forces
{
CenterForce::CenterForce() : Force(0.0001f)
{
}

void CenterForce::beforeAll(std::vector<LabelState> &labels)
{
  Eigen::Vector3f anchorPositionSum;
  for (auto &labelState : labels)
    anchorPositionSum += labelState.anchorPosition;

  if (labels.size() > 0)
    averageCenter = anchorPositionSum / labels.size();
  else
    averageCenter = Eigen::Vector3f(0, 0, 0);
}

Eigen::Vector3f CenterForce::calculate(LabelState &label,
                                       std::vector<LabelState> &labels,
                                       const LabellerFrameData &frameData)
{
  return (label.anchorPosition - averageCenter).normalized() * weight;
}

}  // namespace Forces
