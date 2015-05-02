#ifndef SRC_FORCES_LINES_CROSSING_FORCE_H_

#define SRC_FORCES_LINES_CROSSING_FORCE_H_

#include "./force.h"

namespace Forces
{
/**
 * \brief
 *
 *
 */
class LinesCrossingForce : public Force
{
 public:
  LinesCrossingForce();

  Eigen::Vector2f calculate(LabelState &label, std::vector<LabelState> &labels,
                            const LabellerFrameData &frameData);

 private:
  bool doLinesIntersect(const LabelState &current, const LabelState &other);
  Eigen::Vector2f calculateForce(const LabelState &current,
                                 const LabelState &other);
};
}  // namespace Forces

#endif  // SRC_FORCES_LINES_CROSSING_FORCE_H_
