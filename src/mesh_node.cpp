#include "./mesh_node.h"
#include <string>
#include <Eigen/Eigenvalues>
#include "./mesh.h"
#include "./gl.h"

Eigen::Vector3f project(Eigen::Vector3f vector, Eigen::Vector3f onto)
{
  return vector.dot(onto) * onto;
}

MeshNode::MeshNode(std::string assetFilename, int meshIndex,
                   std::shared_ptr<Mesh> mesh, Eigen::Matrix4f transformation)
  : assetFilename(assetFilename), meshIndex(meshIndex), mesh(mesh),
    transformation(transformation)
{
  std::cout << assetFilename << " " << meshIndex << " " << mesh->vertexCount
            << std::endl;

  auto vertexCount = 4;
  // auto vertexCount = mesh->vertexCount;
  Eigen::MatrixXf data(3, vertexCount);
  /*
  auto positions = mesh->positionData;
  for (int i = 0; i < vertexCount; ++i)
    data.col(i) = Eigen::Vector3f(positions[i * 3], positions[i * 3 + 1],
                                  positions[i * 3 + 2]);
                                  */
  data.col(0) = Eigen::Vector3f(1, 0, 0);
  data.col(1) = Eigen::Vector3f(-1, 0, 0);
  data.col(2) = Eigen::Vector3f(1, 0.1, 0);
  data.col(3) = Eigen::Vector3f(1, 0.2, 0);

  // std::cout << data << std::endl;

  Eigen::Matrix3Xf centered = data.colwise() - data.rowwise().mean();
  std::cout << "centered" << std::endl;
  std::cout << centered << std::endl;

  Eigen::MatrixXf cov = centered * centered.adjoint();
  std::cout << "cov" << std::endl;
  std::cout << cov << std::endl;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);
  Eigen::MatrixXf eigenVectors = eig.eigenvectors();
  std::cout << "eigenvectors" << std::endl;
  std::cout << eigenVectors << std::endl;

  // Gram-Schmidt process
  Eigen::Vector3f axis1 = eigenVectors.row(2).normalized();
  Eigen::Vector3f eigen2 = eigenVectors.row(1);
  Eigen::Vector3f dir2 = eigen2 - project(eigen2, axis1);
  Eigen::Vector3f axis2 = dir2.normalized();
  Eigen::Vector3f axis3 = axis1.cross(axis2);

  Eigen::MatrixXf projectedOnAxis1 = axis1.transpose() * centered;
  Eigen::MatrixXf projectedOnAxis2 = axis2.transpose() * centered;
  Eigen::MatrixXf projectedOnAxis3 = axis3.transpose() * centered;
  std::cout << "Projected onto axis 1" << std::endl;
  std::cout << projectedOnAxis1 << std::endl;
  std::cout << "Projected onto axis 2" << std::endl;
  std::cout << projectedOnAxis2 << std::endl;
  std::cout << "Projected onto axis 3" << std::endl;
  std::cout << projectedOnAxis3 << std::endl;
}

MeshNode::~MeshNode()
{
}

void MeshNode::render(Gl *gl, RenderData renderData)
{
  renderData.modelMatrix = transformation;
  mesh->render(gl, renderData);
}

Eigen::Matrix4f MeshNode::getTransformation()
{
  return this->transformation;
}

void MeshNode::setTransformation(Eigen::Matrix4f transformation)
{
  this->transformation = transformation;
}
