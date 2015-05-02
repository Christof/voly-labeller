#ifndef SRC_FORCES_LABEL_STATE_H_

#define SRC_FORCES_LABEL_STATE_H_

#include <Eigen/Core>
#include <string>

namespace Forces
{
/**
 * \brief
 *
 *
 */
class LabelState
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LabelState(int id, std::string text, Eigen::Vector3f anchorPosition);

  const int id;
  const Eigen::Vector3f anchorPosition;
  Eigen::Vector3f labelPosition;

  Eigen::Vector2f anchorPosition2D;
  Eigen::Vector2f labelPosition2D;
  float labelPositionDepth;

  const std::string text;
};
}  // namespace Forces

#endif  // SRC_FORCES_LABEL_STATE_H_
