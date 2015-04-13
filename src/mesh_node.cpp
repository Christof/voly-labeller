#include "./mesh_node.h"
#include "./mesh.h"
#include "./gl.h"

MeshNode::MeshNode(std::string assetFilename, int meshIndex,
                   std::shared_ptr<Mesh> mesh, Eigen::Matrix4f transformation)
  : assetFilename(assetFilename), meshIndex(meshIndex), mesh(mesh),
    transformation(transformation)
{
}

MeshNode::~MeshNode()
{
}

void MeshNode::render(Gl *gl, const RenderData &renderData)
{
  mesh->render(gl, renderData.projectionMatrix, renderData.viewMatrix);
}

Eigen::Matrix4f MeshNode::getTransformation()
{
  return this->transformation;
}

void MeshNode::setTransformation(Eigen::Matrix4f transformation)
{
  this->transformation = transformation;
}
