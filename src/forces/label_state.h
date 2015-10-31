#ifndef SRC_FORCES_LABEL_STATE_H_

#define SRC_FORCES_LABEL_STATE_H_

#include <Eigen/Core>
#include <string>
#include <map>

namespace Forces
{

class Force;
/**
 * \brief Encapsulates state for a label necessary for the simulation
 *
 * This consists of label and anchor positions in 2D and 3D, as well as
 * the label's size, text and its id. All 2D data is given in normalized
 * device coordinates.
 *
 * It also stores the last force values for debugging purposes.
 */
class LabelState
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LabelState(int id, std::string text, Eigen::Vector3f anchorPosition,
             Eigen::Vector2f size);

  int id;
  Eigen::Vector3f anchorPosition;
  Eigen::Vector3f labelPosition;
  Eigen::Vector2f size;

  Eigen::Vector2f anchorPosition2D;
  Eigen::Vector2f labelPosition2D;
  float labelPositionDepth;

  std::string text;

  std::map<Force *, Eigen::Vector2f> forces;
};
}  // namespace Forces

#endif  // SRC_FORCES_LABEL_STATE_H_
