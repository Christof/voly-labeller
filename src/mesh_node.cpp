#include "./mesh_node.h"
#include "./mesh.h"

MeshNode::MeshNode(std::shared_ptr<Mesh> mesh, Eigen::Matrix4f transformation)
  : mesh(mesh), transformation(transformation)
{
}

MeshNode::~MeshNode()
{
}

void MeshNode::render(const RenderData &renderData)
{
  mesh->render(renderData.projectionMatrix, renderData.viewMatrix);
}

Eigen::Matrix4f MeshNode::getTransformation()
{
  return this->transformation;
}

void MeshNode::setTransformation(Eigen::Matrix4f transformation)
{
  this->transformation = transformation;
}
