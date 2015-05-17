#ifndef SRC_FORCES_FORCE_H_

#define SRC_FORCES_FORCE_H_

#include <Eigen/Core>
#include <vector>
#include <string>
#include "./labeller_frame_data.h"

namespace Forces
{
class LabelState;

/**
 * \brief Base class for all forces, which handles meta data and weights
 *
 * The force can be disabled with Force::isEnabled.
 *
 * The influence can be changed by setting Force::weight.
 */
class Force
{
 public:
  Force(std::string name, float weight);
  ~Force() = default;

  virtual void beforeAll(std::vector<LabelState> &labels);

  Eigen::Vector2f calculateForce(LabelState &label,
                                 std::vector<LabelState> &labels,
                                 const LabellerFrameData &frameData);

  const Eigen::Vector3f color;
  const std::string name;
  bool isEnabled = true;
  float weight;

 protected:
  virtual Eigen::Vector2f calculate(LabelState &label,
                                    std::vector<LabelState> &labels) = 0;
};
}  // namespace Forces

#endif  // SRC_FORCES_FORCE_H_
