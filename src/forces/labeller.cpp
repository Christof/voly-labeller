#include "./labeller.h"
#include <map>
#include <string>
#include "./center_force.h"

namespace Forces
{
void Labeller::addLabel(int id, std::string text,
                        Eigen::Vector3f anchorPosition)
{
  labels.push_back(LabelState(id, text, anchorPosition));

  forces.push_back(std::unique_ptr<CenterForce>(new CenterForce()));
}

std::map<int, Eigen::Vector3f> Labeller::update(double frameTime)
{
  std::map<int, Eigen::Vector3f> positions;

  for (auto &force : forces)
    force->beforeAll(labels);

  for (auto &label : labels)
  {
    auto forceOnLabel = Eigen::Vector3f(0, 0, 0);
    for (auto &force : forces)
      forceOnLabel += force->calculate(label, labels);

    label.labelPosition += forceOnLabel * frameTime;

    positions[label.id] = label.labelPosition;
  }

  return positions;
}
}  // namespace Forces
