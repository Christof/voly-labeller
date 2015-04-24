#include "./mesh_node.h"
#include <string>
#include "./mesh.h"
#include "./gl.h"

MeshNode::MeshNode(std::string assetFilename, int meshIndex,
                   std::shared_ptr<Mesh> mesh, Eigen::Matrix4f transformation)
  : assetFilename(assetFilename), meshIndex(meshIndex), mesh(mesh),
    transformation(transformation)
{
  auto obb = *mesh->obb;
  obbVis = std::make_shared<Connector>(std::vector<Eigen::Vector3f>{
    obb.corners[0], obb.corners[1], obb.corners[1], obb.corners[2],
    obb.corners[2], obb.corners[3], obb.corners[3], obb.corners[0],
    obb.corners[4], obb.corners[5], obb.corners[5], obb.corners[6],
    obb.corners[6], obb.corners[7], obb.corners[7], obb.corners[4],
    obb.corners[0], obb.corners[4], obb.corners[1], obb.corners[5],
    obb.corners[2], obb.corners[6], obb.corners[3], obb.corners[7]
  });
  obbVis->color = Eigen::Vector4f(meshIndex, 0.5f, 0.5f, 1);
  obbVis->lineWidth = 1.0f;
}

MeshNode::~MeshNode()
{
}

void MeshNode::render(Gl *gl, RenderData renderData)
{
  renderData.modelMatrix = transformation;
  mesh->render(gl, renderData);

  obbVis->render(gl, renderData);
}

Eigen::Matrix4f MeshNode::getTransformation()
{
  return this->transformation;
}

void MeshNode::setTransformation(Eigen::Matrix4f transformation)
{
  this->transformation = transformation;
}
