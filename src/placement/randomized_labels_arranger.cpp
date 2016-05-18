#include "./randomized_labels_arranger.h"
#include <vector>
#include <algorithm>

namespace Placement
{

std::vector<Label>
RandomizedLabelsArranger::getArrangement(const LabellerFrameData &frameData,
                                        std::shared_ptr<LabelsContainer> labels)
{
  auto vector = labels->getLabels();

  std::random_shuffle(vector.begin(), vector.end());

  return vector;
}

}  // namespace Placement
