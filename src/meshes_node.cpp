#include "./meshes_node.h"
#include <string>
#include "./importer.h"
#include "./graphics/gl.h"

MeshesNode::MeshesNode(std::string assetFilename,
                       Eigen::Matrix4f transformation)
  : assetFilename(assetFilename), transformation(transformation)
{
  // obb = mesh->obb * transformation;
}

MeshesNode::~MeshesNode()
{
}

void MeshesNode::render(Graphics::Gl *gl,
                        std::shared_ptr<Graphics::Managers> managers,
                        RenderData renderData)
{
  if (meshes.size() == 0)
    loadMeshes();

  if (!isVisible)
    return;

  for (auto &mesh : meshes)
  {
    renderData.modelMatrix = transformation;
    mesh->render(gl, managers, renderData);
  }
}

Eigen::Matrix4f MeshesNode::getTransformation()
{
  return this->transformation;
}

void MeshesNode::setTransformation(Eigen::Matrix4f transformation)
{
  this->transformation = transformation;
}

void MeshesNode::loadMeshes()
{
  Importer importer;

  meshes = importer.importAll(assetFilename);
  for (size_t index = 0; index < meshes.size(); ++index)
  {
    transformations.push_back(
        importer.getTransformationFor(assetFilename, index));
  }
}

