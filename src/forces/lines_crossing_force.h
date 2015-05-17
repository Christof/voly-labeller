#ifndef SRC_FORCES_LINES_CROSSING_FORCE_H_

#define SRC_FORCES_LINES_CROSSING_FORCE_H_

#include <vector>
#include "./force.h"

namespace Forces
{
/**
 * \brief If lines between anchors and labels cross it pushes the labels
 * away in a perpendicular fashion
 *
 */
class LinesCrossingForce : public Force
{
 public:
  LinesCrossingForce();

  Eigen::Vector2f calculate(LabelState &label, std::vector<LabelState> &labels);

 private:
  bool doLinesIntersect(const LabelState &current, const LabelState &other);
  Eigen::Vector2f calculateCrossingForce(const LabelState &current,
                                 const LabelState &other);
};
}  // namespace Forces

#endif  // SRC_FORCES_LINES_CROSSING_FORCE_H_
