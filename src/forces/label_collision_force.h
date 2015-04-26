#ifndef SRC_FORCES_LABEL_COLLISION_FORCE_H_

#define SRC_FORCES_LABEL_COLLISION_FORCE_H_

#include "./force.h"

namespace Forces
{
/**
 * \brief
 *
 *
 */
class LabelCollisionForce : public Force
{
 public:
  LabelCollisionForce();

  Eigen::Vector3f calculate(LabelState &label, std::vector<LabelState> &labels);
};
}  // namespace Forces

#endif  // SRC_FORCES_LABEL_COLLISION_FORCE_H_
