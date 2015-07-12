#include "./mesh_node.h"
#include <string>
#include "./graphics/gl.h"

MeshNode::MeshNode(std::string assetFilename, int meshIndex,
                   std::shared_ptr<Graphics::Mesh> mesh,
                   Eigen::Matrix4f transformation)
  : assetFilename(assetFilename), meshIndex(meshIndex), mesh(mesh),
    transformation(transformation)
{
}

MeshNode::~MeshNode()
{
}

std::shared_ptr<Math::Obb> MeshNode::getObb()
{
  return mesh->obb;
}

void MeshNode::render(Graphics::Gl *gl, RenderData renderData)
{
  renderData.modelMatrix = transformation;
  mesh->render(gl, haBuffer, renderData);
}

Eigen::Matrix4f MeshNode::getTransformation()
{
  return this->transformation;
}

void MeshNode::setTransformation(Eigen::Matrix4f transformation)
{
  this->transformation = transformation;
}

