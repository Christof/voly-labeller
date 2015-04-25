#ifndef SRC_FORCES_CENTER_FORCE_H_

#define SRC_FORCES_CENTER_FORCE_H_

#include <Eigen/Core>
#include <vector>

namespace Forces
{

class LabelState;

/**
 * \brief
 *
 *
 */
class CenterForce
{
 public:
  CenterForce() = default;

  void beforeAll(std::vector<LabelState> &labels);
  Eigen::Vector3f calculate(LabelState &label, std::vector<LabelState> &labels);

 private:
  Eigen::Vector3f averageCenter;
  float weight = 0.0001f;
};
}  // namespace Forces

#endif  // SRC_FORCES_CENTER_FORCE_H_
