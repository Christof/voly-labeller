#ifndef SRC_FORCES_LABEL_COLLISION_FORCE_H_

#define SRC_FORCES_LABEL_COLLISION_FORCE_H_

#include <vector>
#include "./force.h"

namespace Forces
{
/**
 * \brief If two labels collide they are pushed away from each other
 *
 */
class LabelCollisionForce : public Force
{
 public:
  LabelCollisionForce();

 protected:
  Eigen::Vector2f calculate(LabelState &label, std::vector<LabelState> &labels);
};
}  // namespace Forces

#endif  // SRC_FORCES_LABEL_COLLISION_FORCE_H_
