#include "./force.h"
#include <vector>
#include <string>
#include "./label_state.h"

namespace Forces
{

Q_LOGGING_CATEGORY(forcesChan, "Forces");

Force::Force(std::string name, float weight)
  : color(Eigen::Vector3f::Random().cwiseAbs()), name(name), weight(weight)
{
}

void Force::beforeAll(std::vector<LabelState> &labels)
{
}

Eigen::Vector2f Force::calculateForce(LabelState &label,
                                      std::vector<LabelState> &labels,
                                      const LabellerFrameData &frameData)
{
  if (!isEnabled)
  {
    label.forces[this] = Eigen::Vector2f::Zero();
    return Eigen::Vector2f::Zero();
  }

  Eigen::Vector2f force =
      weight * frameData.frameTime * calculate(label, labels);

  label.forces[this] = force;

  return force;
}

}  // namespace Forces

