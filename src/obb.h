#ifndef SRC_OBB_H_

#define SRC_OBB_H_

#include <Eigen/Core>

/**
 * \brief
 *
 *
 */
class Obb
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Obb() = default;
  Obb(Eigen::MatrixXf points);

  Eigen::Vector3f corners[8];

 private:
  Eigen::Matrix3f axes;

  Eigen::Vector3f halfWidths;

  Eigen::Vector3f center;

  void calculateCenterAndHalfWidhts(Eigen::MatrixXf &onAxes);
  void calculateCorners();
  Eigen::Vector3f getCornerWidths(float axis1Sign, float axis2Sign, float axis3Sign);
};

#endif  // SRC_OBB_H_
