#ifndef SRC_FORCES_LABELLER_FRAME_DATA_H_

#define SRC_FORCES_LABELLER_FRAME_DATA_H_

#include "../math/eigen.h"

namespace Forces
{
/**
 * \brief
 *
 *
 */
class LabellerFrameData
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LabellerFrameData(double frameTime, Eigen::Matrix4f projection,
                    Eigen::Matrix4f view)
    : frameTime(frameTime), projection(projection), view(view),
      viewProjection(projection * view)
  {
  }

  const double frameTime;
  const Eigen::Matrix4f projection;
  const Eigen::Matrix4f view;
  const Eigen::Matrix4f viewProjection;

  Eigen::Vector3f project(Eigen::Vector3f vector) const
  {
    Eigen::Vector4f projected = mul(viewProjection, vector);

    return projected.head<3>() / projected.w();
  }
};
}  // namespace Forces

#endif  // SRC_FORCES_LABELLER_FRAME_DATA_H_
