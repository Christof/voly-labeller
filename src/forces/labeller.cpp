#include "./labeller.h"

namespace Forces
{
void Labeller::addLabel(int id, std::string text,
                        Eigen::Vector3f anchorPosition)
{
  labels.push_back(LabelState(id, text, anchorPosition));
}

std::map<int, Eigen::Vector3f> Labeller::update(double frameTime)
{
  std::map<int, Eigen::Vector3f> positions;

  centerForce.beforeAll(labels);

  for (auto &label : labels)
  {
    auto force = centerForce.calculate(label, labels);
    label.labelPosition += force * frameTime;

    positions[label.id] = label.labelPosition;
  }

  return positions;
}
}  // namespace Forces
