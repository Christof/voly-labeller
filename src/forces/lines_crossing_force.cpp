#include "./lines_crossing_force.h"
#include "../collision.h"
#include "./label_state.h"

namespace Forces
{
Eigen::Vector2f
LinesCrossingForce::calculate(LabelState &label,
                              std::vector<LabelState> &labels,
                              const LabellerFrameData &frameData)
{
  Eigen::Vector2f result(0, 0);

  for (auto &otherLabel : labels)
  {
    if (otherLabel.id == label.id)
      continue;

    if (doLinesIntersect(label, otherLabel))
      result += calculateForce(label, otherLabel);
  }

  return result;
}

bool LinesCrossingForce::doLinesIntersect(const LabelState &current,
                                          const LabelState &other)
{
  return test2DSegmentSegment(current.labelPosition2D, current.anchorPosition2D,
                              other.labelPosition2D, other.anchorPosition2D);
}

Eigen::Vector2f LinesCrossingForce::calculateForce(const LabelState &current,
                                                   const LabelState &ohter)
{
  return Eigen::Vector2f(0, 0);
}
}  // namespace Forces
