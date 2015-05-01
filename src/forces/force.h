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
  Force(float weight) : weight(weight)
  {
  }

  virtual void beforeAll(std::vector<LabelState> &labels)
  {
  }

  virtual Eigen::Vector2f calculate(LabelState &label,
                                    std::vector<LabelState> &labels,
                                    const LabellerFrameData &frameData) = 0;

 protected:
  float weight;
};
}  // namespace Forces

#endif  // SRC_FORCES_FORCE_H_
