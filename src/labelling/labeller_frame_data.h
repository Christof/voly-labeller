#ifndef SRC_LABELLING_LABELLER_FRAME_DATA_H_

#define SRC_LABELLING_LABELLER_FRAME_DATA_H_

#include "../math/eigen.h"

/**
 * \brief Data which changes every frame and is passed on to the labeller
 * and then to the forces
 *
 * It consists of camera matrices and the frame time.
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

  Eigen::Vector2f project2d(Eigen::Vector3f vector) const
  {
    return project(vector).head<2>();
  }
};

#endif  // SRC_LABELLING_LABELLER_FRAME_DATA_H_
