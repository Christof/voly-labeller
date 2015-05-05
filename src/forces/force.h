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
 * \brief
 *
 *
 */
class Force
{
 public:
  Force(std::string name, float weight);

  virtual void beforeAll(std::vector<LabelState> &labels);

  Eigen::Vector2f calculateForce(LabelState &label,
                                 std::vector<LabelState> &labels,
                                 const LabellerFrameData &frameData);

  const Eigen::Vector3f color;
  const std::string name;
  bool isEnabled = true;

 protected:
  virtual Eigen::Vector2f calculate(LabelState &label,
                                    std::vector<LabelState> &labels) = 0;

 private:
  float weight;
};
}  // namespace Forces

#endif  // SRC_FORCES_FORCE_H_
