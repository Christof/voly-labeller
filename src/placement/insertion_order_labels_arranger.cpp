#include "./insertion_order_labels_arranger.h"
#include <vector>

namespace Placement
{

std::vector<Label> InsertionOrderLabelsArranger::getArrangement(
    const LabellerFrameData &frameData, std::shared_ptr<LabelsContainer> labels)
{
  return labels->getLabels();
}

}  // namespace Placement
