#ifndef SRC_PLACEMENT_RANDOMIZED_LABELS_ARRANGER_H_

#define SRC_PLACEMENT_RANDOMIZED_LABELS_ARRANGER_H_

#include "./labels_arranger.h"

namespace Placement
{

/**
 * \brief Returns the labels in an randomized order
 *
 */
class RandomizedLabelsArranger : public LabelsArranger
{
 public:
  RandomizedLabelsArranger() = default;

  virtual std::vector<Label>
  getArrangement(const LabellerFrameData &frameData,
                 std::shared_ptr<LabelsContainer> labels);
};

}  // namespace Placement

#endif  // SRC_PLACEMENT_RANDOMIZED_LABELS_ARRANGER_H_
