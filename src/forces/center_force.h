#ifndef SRC_FORCES_CENTER_FORCE_H_

#define SRC_FORCES_CENTER_FORCE_H_

#include <Eigen/Core>
#include <vector>
#include "./force.h"

namespace Forces
{

class LabelState;

/**
 * \brief
 *
 *
 */
class CenterForce : public Force
{
 public:
  CenterForce();

  void beforeAll(std::vector<LabelState> &labels);
  Eigen::Vector2f calculate(LabelState &label, std::vector<LabelState> &labels,
                            const LabellerFrameData &frameData);

 private:
  Eigen::Vector2f averageCenter;
};
}  // namespace Forces

#endif  // SRC_FORCES_CENTER_FORCE_H_
