#include "./lines_crossing_force.h"
#include <vector>
#include "../collision.h"
#include "./label_state.h"
#include "../eigen_qdebug.h"

namespace Forces
{
LinesCrossingForce::LinesCrossingForce() : Force(1)
{
}

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

  return result * frameData.frameTime;
}

bool LinesCrossingForce::doLinesIntersect(const LabelState &current,
                                          const LabelState &other)
{
  return test2DSegmentSegment(current.labelPosition2D, current.anchorPosition2D,
                              other.labelPosition2D, other.anchorPosition2D);
}

Eigen::Vector2f LinesCrossingForce::calculateForce(const LabelState &current,
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
  qDebug() << "Lines crossing " << current.text.c_str() << " force:" << force;

  return force;
}
}  // namespace Forces
