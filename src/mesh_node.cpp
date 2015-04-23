#include "./mesh_node.h"
#include <string>
#include "./mesh.h"
#include "./gl.h"

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

  obb = Obb(data);
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
