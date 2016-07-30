#include "./mesh_node.h"
#include <string>
#include "./graphics/gl.h"

MeshNode::MeshNode(std::string assetFilename, int meshIndex,
                   std::shared_ptr<Graphics::Mesh> mesh,
                   Eigen::Matrix4f transformation)
  : assetFilename(assetFilename), meshIndex(meshIndex), mesh(mesh),
    transformation(transformation)
{
  obb = mesh->obb * transformation;
}

MeshNode::~MeshNode()
{
}

void MeshNode::render(Graphics::Gl *gl,
                      std::shared_ptr<Graphics::Managers> managers,
                      RenderData renderData)
{
  if (!isVisible)
    return;

  renderData.modelMatrix = transformation;
  mesh->render(gl, managers, renderData);
}

Eigen::Matrix4f MeshNode::getTransformation()
{
  return this->transformation;
}

void MeshNode::setTransformation(Eigen::Matrix4f transformation)
{
  this->transformation = transformation;
}

