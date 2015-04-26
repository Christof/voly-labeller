#ifndef SRC_FORCES_ANCHOR_FORCE_H_

#define SRC_FORCES_ANCHOR_FORCE_H_

#include "./force.h"

namespace Forces
{

class LabelState;

/**
 * \brief
 *
 *
 */
class AnchorForce : public Force
{
 public:
  AnchorForce();

  Eigen::Vector3f calculate(LabelState &label, std::vector<LabelState> &labels);
};
}  // namespace Forces

#endif  // SRC_FORCES_ANCHOR_FORCE_H_
