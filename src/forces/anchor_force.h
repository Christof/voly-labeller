#ifndef SRC_FORCES_ANCHOR_FORCE_H_

#define SRC_FORCES_ANCHOR_FORCE_H_

#include <vector>
#include "./force.h"

namespace Forces
{

class LabelState;

/**
 * \brief Pulls the label towards the anchor or pushes it away if it
 * is too close
 *
 */
class AnchorForce : public Force
{
 public:
  AnchorForce();

  Eigen::Vector2f calculate(LabelState &label, std::vector<LabelState> &labels);
};
}  // namespace Forces

#endif  // SRC_FORCES_ANCHOR_FORCE_H_
