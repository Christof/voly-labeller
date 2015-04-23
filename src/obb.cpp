#include "./obb.h"
#include <Eigen/Eigenvalues>
#include <iostream>

Obb::Obb(Eigen::MatrixXf points)
{
  Eigen::Matrix3Xf centered = points.colwise() - points.rowwise().mean();

  Eigen::MatrixXf cov = centered * centered.adjoint();
  std::cout << "cov" << std::endl;
  std::cout << cov << std::endl;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);
  Eigen::MatrixXf eigenVectors = eig.eigenvectors();
  std::cout << "eigenvectors" << std::endl;
  std::cout << eigenVectors << std::endl;

  // Gram-Schmidt process
  axis1 = eigenVectors.row(2).normalized();
  auto eigen2 = eigenVectors.row(1);
  auto dir2 = eigen2 - eigen2.dot(axis1) * eigen2;
  axis2 = dir2.normalized();
  axis3 = axis1.cross(axis2);

  Eigen::MatrixXf onAxis1 = axis1.transpose() * points;
  Eigen::MatrixXf onAxis2 = axis2.transpose() * points;
  Eigen::MatrixXf onAxis3 = axis3.transpose() * points;
  std::cout << "Projected onto axis 1" << std::endl;
  std::cout << onAxis1 << std::endl;
  std::cout << "Projected onto axis 2" << std::endl;
  std::cout << onAxis2 << std::endl;
  std::cout << "Projected onto axis 3" << std::endl;
  std::cout << onAxis3 << std::endl;

  auto upper = Eigen::Vector3f(onAxis1.maxCoeff(), onAxis2.maxCoeff(),
                               onAxis3.maxCoeff());
  auto lower = Eigen::Vector3f(onAxis1.minCoeff(), onAxis2.minCoeff(),
                               onAxis3.minCoeff());

  halfWidths = 0.5f * (lower + upper);

  center =
      halfWidths.x() * axis1 + halfWidths.y() * axis2 + halfWidths.z() * axis3;

}
