#ifndef SRC_PLACEMENT_LABELLER_H_

#define SRC_PLACEMENT_LABELLER_H_

#include <memory>
#include "../labelling/labels.h"

namespace Placement
{

/**
 * \brief
 *
 *
 */
class Labeller
{
 public:
  explicit Labeller(std::shared_ptr<Labels> labels);

 private:
  std::shared_ptr<Labels> labels;
};

}  // namespace Placement
#endif  // SRC_PLACEMENT_LABELLER_H_
