#include "./lines_crossing_force.h"
#include <vector>
#include "../math/collision.h"
#include "./label_state.h"
#include "../eigen_qdebug.h"

namespace Forces
{
LinesCrossingForce::LinesCrossingForce() : Force("Lines Crossing", 1)
{
}

Eigen::Vector2f LinesCrossingForce::calculate(LabelState &label,
                                              std::vector<LabelState> &labels)
{
  Eigen::Vector2f result(0, 0);

  for (auto &otherLabel : labels)
  {
    if (otherLabel.id == label.id)
      continue;

    if (doLinesIntersect(label, otherLabel))
      result += calculateCrossingForce(label, otherLabel);
  }

  return result;
}

bool LinesCrossingForce::doLinesIntersect(const LabelState &current,
                                          const LabelState &other)
{
  return Math::test2DSegmentSegment(
      current.labelPosition2D, current.anchorPosition2D, other.labelPosition2D,
      other.anchorPosition2D);
}

Eigen::Vector2f
LinesCrossingForce::calculateCrossingForce(const LabelState &current,
                                           const LabelState &other)
{
  Eigen::Vector2f direction = other.labelPosition2D - current.labelPosition2D;
  direction.normalize();

  Eigen::Vector2f orthoDir = current.labelPosition2D - current.anchorPosition2D;
  float temp = orthoDir.y();
  orthoDir.y() = orthoDir.x() * -1.0f;
  orthoDir.x() = temp;
  orthoDir.normalize();

  Eigen::Vector2f force = direction.dot(orthoDir) * orthoDir;
  force.normalize();
  qCDebug(forcesChan) << "Lines crossing " << current.text.c_str()
                      << " force:" << force;

  return force;
}
}  // namespace Forces
