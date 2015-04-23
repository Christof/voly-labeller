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
  Eigen::Vector3f axis1;
  Eigen::Vector3f axis2;
  Eigen::Vector3f axis3;

  Eigen::Vector3f halfWidths;

  Eigen::Vector3f center;

};

#endif  // SRC_OBB_H_
