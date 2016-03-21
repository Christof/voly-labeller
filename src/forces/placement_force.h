#ifndef SRC_FORCES_PLACEMENT_FORCE_H_

#define SRC_FORCES_PLACEMENT_FORCE_H_

#include <Eigen/Core>
#include <vector>
#include "./force.h"

namespace Forces
{

class LabelState;

/**
 * \brief Pulls the label towards the position calculated by the placement
 * algorithm
 */
class PlacementForce : public Force
{
 public:
  PlacementForce();

 protected:
  virtual Eigen::Vector2f calculate(LabelState &label,
                                    std::vector<LabelState> &labels);
};

}  // namespace Forces

#endif  // SRC_FORCES_PLACEMENT_FORCE_H_
