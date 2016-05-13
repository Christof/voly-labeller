#include "./labels_container.h"
#include <vector>

void LabelsContainer::add(Label label)
{
  labels[label.id] = label;
}

Label LabelsContainer::getById(int id)
{
  if (labels.count(id))
    return labels[id];

  return Label();
}

int LabelsContainer::count()
{
  return static_cast<int>(labels.size());
}

void LabelsContainer::clear()
{
  labels.clear();
}

std::vector<Label> LabelsContainer::getLabels()
{
  std::vector<Label> result;
  for (auto &pair : labels)
    result.push_back(pair.second);

  return result;
}
