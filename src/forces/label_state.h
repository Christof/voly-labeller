#ifndef SRC_FORCES_LABEL_STATE_H_

#define SRC_FORCES_LABEL_STATE_H_

#include <Eigen/Core>
#include <string>
#include <map>

namespace Forces
{

class Force;
/**
 * \brief
 *
 *
 */
class LabelState
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LabelState(int id, std::string text, Eigen::Vector3f anchorPosition,
             Eigen::Vector2f size);

  const int id;
  const Eigen::Vector3f anchorPosition;
  Eigen::Vector3f labelPosition;
  Eigen::Vector2f size;

  Eigen::Vector2f anchorPosition2D;
  Eigen::Vector2f labelPosition2D;
  float labelPositionDepth;

  const std::string text;

  std::map<Force *, Eigen::Vector2f> forces;
};
}  // namespace Forces

#endif  // SRC_FORCES_LABEL_STATE_H_
