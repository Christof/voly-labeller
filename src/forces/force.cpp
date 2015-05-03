#include "./force.h"
#include <vector>
#include "./label_state.h"

namespace Forces
{

Force::Force(float weight)
  : color(Eigen::Vector3f::Random()), weight(weight)
{
}

void Force::beforeAll(std::vector<LabelState> &labels)
{
}

Eigen::Vector2f Force::calculateForce(LabelState &label,
                                      std::vector<LabelState> &labels,
                                      const LabellerFrameData &frameData)
{
  Eigen::Vector2f force = weight * calculate(label, labels, frameData);

  label.forces[this] = force;

  return force;
}

}  // namespace Forces

