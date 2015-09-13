#include "./mesh_node.h"
#include <string>
#include "./graphics/gl.h"

MeshNode::MeshNode(std::string assetFilename, int meshIndex,
                   std::shared_ptr<Graphics::Mesh> mesh,
                   Eigen::Matrix4f transformation)
  : assetFilename(assetFilename), meshIndex(meshIndex), mesh(mesh),
    transformation(transformation)
{
  obb = std::make_shared<Math::Obb>((*mesh->obb.get()) * transformation);
}

MeshNode::~MeshNode()
{
}

std::shared_ptr<Math::Obb> MeshNode::getObb()
{
  return obb;
}

void MeshNode::render(Graphics::Gl *gl, RenderData renderData)
{
  renderData.modelMatrix = transformation;
  mesh->render(gl, objectManager, textureManager, shaderManager, renderData);
}

Eigen::Matrix4f MeshNode::getTransformation()
{
  return this->transformation;
}

void MeshNode::setTransformation(Eigen::Matrix4f transformation)
{
  this->transformation = transformation;
}

