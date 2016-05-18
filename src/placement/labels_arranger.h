#ifndef SRC_PLACEMENT_LABELS_ARRANGER_H_

#define SRC_PLACEMENT_LABELS_ARRANGER_H_

#include <memory>
#include <vector>
#include "../labelling/labeller_frame_data.h"
#include "../labelling/labels_container.h"

namespace Placement
{

/**
 * \brief Interface for classes which determine the insertion order of labels
 *
 * The labels are given by a LabelsContainer
 * and a vector of Label\s is returned.
 */
class LabelsArranger
{
 public:
  virtual std::vector<Label>
  getArrangement(const LabellerFrameData &frameData,
                 std::shared_ptr<LabelsContainer> labels) = 0;
};

}  // namespace Placement

#endif  // SRC_PLACEMENT_LABELS_ARRANGER_H_
