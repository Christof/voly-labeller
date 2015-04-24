#include "./obb.h"
#include <Eigen/Eigenvalues>
#include <iostream>

Obb::Obb(Eigen::MatrixXf points)
{
  Eigen::Matrix3Xf centered = points.colwise() - points.rowwise().mean();

  Eigen::MatrixXf cov = centered * centered.adjoint();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);
  Eigen::MatrixXf eigenVectors = eig.eigenvectors();

  axes.row(2) = eigenVectors.row(0).normalized();
  axes.row(1) = eigenVectors.row(1).normalized();
  axes.row(0) = eigenVectors.row(2).normalized();

  Eigen::MatrixXf onAxes = axes * points;

  calculateCenterAndHalfWidhts(onAxes);

  calculateCorners();
}

void Obb::calculateCenterAndHalfWidhts(Eigen::MatrixXf &onAxes)
{
  float max = std::numeric_limits<float>::max();
  Eigen::Vector3f upper(-max, -max, -max);
  Eigen::Vector3f lower(max, max, max);

  for (int i = 0; i < onAxes.cols(); ++i)
  {
    Eigen::Vector3f vector = onAxes.col(i);

    if (vector.x() > upper.x())
      upper.x() = vector.x();
    if (vector.y() > upper.y())
      upper.y() = vector.y();
    if (vector.z() > upper.z())
      upper.z() = vector.z();

    if (vector.x() < lower.x())
      lower.x() = vector.x();
    if (vector.y() < lower.y())
      lower.y() = vector.y();
    if (vector.z() < lower.z())
      lower.z() = vector.z();
  }

  halfWidths = 0.5f * (upper - lower);

  auto halfSum = 0.5f * (upper + lower);
  center = axes.transpose() * halfSum;
}

void Obb::calculateCorners()
{
  corners[0] = center - axes * getCornerWidths(1, 1, 1);
  corners[1] = center + axes * getCornerWidths(1, -1, -1);
  corners[2] = center + axes * getCornerWidths(1, -1, 1);
  corners[3] = center + axes * getCornerWidths(-1, -1, 1);
  corners[4] = center + axes * getCornerWidths(-1, 1, -1);
  corners[5] = center + axes * getCornerWidths(1, 1, -1);
  corners[6] = center + axes * getCornerWidths(1, 1, 1);
  corners[7] = center + axes * getCornerWidths(-1, 1, 1);
}

Eigen::Vector3f Obb::getCornerWidths(float axis1Sign, float axis2Sign,
                                     float axis3Sign)
{
  return halfWidths.cwiseProduct(
      Eigen::Vector3f(axis1Sign, axis2Sign, axis3Sign));
}

