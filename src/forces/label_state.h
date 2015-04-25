#ifndef SRC_FORCES_LABEL_STATE_H_

#define SRC_FORCES_LABEL_STATE_H_

#include <Eigen/Core>

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
  Eigen::Vector3f anchorPosition;

 private:
  std::string text;

  Eigen::Vector2f anchorPosition2D;
};
}  // namespace Forces

#endif  // SRC_FORCES_LABEL_STATE_H_
