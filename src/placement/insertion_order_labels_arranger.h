#ifndef SRC_PLACEMENT_INSERTION_ORDER_LABELS_ARRANGER_H_

#define SRC_PLACEMENT_INSERTION_ORDER_LABELS_ARRANGER_H_

#include "./labels_arranger.h"

namespace Placement
{

/**
 * \brief Returns the labels by insertion order
 *
 */
class InsertionOrderLabelsArranger : public LabelsArranger
{
 public:
  InsertionOrderLabelsArranger() = default;

  virtual std::vector<Label>
  getArrangement(const LabellerFrameData &frameData,
                 std::shared_ptr<LabelsContainer> labels);
};

}  // namespace Placement

#endif  // SRC_PLACEMENT_INSERTION_ORDER_LABELS_ARRANGER_H_
