#ifndef SRC_FORCES_FORCE_H_

#define SRC_FORCES_FORCE_H_

#include <Eigen/Core>
#include <vector>
#include "./labeller_frame_data.h"

namespace Forces
{
class LabelState;

/**
 * \brief
 *
 *
 */
class Force
{
 public:
  Force(float weight);

  virtual void beforeAll(std::vector<LabelState> &labels);

  Eigen::Vector2f calculateForce(LabelState &label,
                                 std::vector<LabelState> &labels,
                                 const LabellerFrameData &frameData);

 protected:
  float weight;

  virtual Eigen::Vector2f calculate(LabelState &label,
                                    std::vector<LabelState> &labels,
                                    const LabellerFrameData &frameData) = 0;
};
}  // namespace Forces

#endif  // SRC_FORCES_FORCE_H_
