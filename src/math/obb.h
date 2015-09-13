#ifndef SRC_MATH_OBB_H_

#define SRC_MATH_OBB_H_

#include <Eigen/Core>

namespace Math
{

/**
 * \brief Oriented bounding box
 *
 * The algorithm to create it used based on Gottschalk, S. (2000).
 * Collision Queries using Oriented Bounding Boxes. Retrieved from
 * http://www.mechcore.net/files/docs/alg/gottschalk00collision.pdf
 */
class Obb
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Obb() = default;
  explicit Obb(Eigen::MatrixXf points);
  Obb(Eigen::Vector3f center, Eigen::Vector3f halfWidths, Eigen::Matrix3f axes);

  Eigen::Vector3f corners[8];

  Eigen::Vector3f getCenter();
  Eigen::Vector3f getHalfWidths();

  Obb &operator*=(const Eigen::Matrix4f &rhs);
  Obb operator*(const Eigen::Matrix4f &rhs);

 private:
  Eigen::Matrix3f axes;

  Eigen::Vector3f halfWidths;

  Eigen::Vector3f center;

  void calculateCenterAndHalfWidhts(Eigen::MatrixXf &onAxes);
  void calculateCorners();
  Eigen::Vector3f getCornerWidths(float axis1Sign, float axis2Sign,
                                  float axis3Sign);
};

}  // namespace Math
#endif  // SRC_MATH_OBB_H_
