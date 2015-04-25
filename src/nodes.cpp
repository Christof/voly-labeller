#include "./nodes.h"
#include <QDebug>
#include <string>
#include <vector>
#include "./utils/persister.h"
#include "./importer.h"
#include "./gl.h"
#include "./mesh_node.h"
#include "./label_node.h"
#include "./obb_node.h"
#include "./coordinate_system_node.h"

Nodes::Nodes()
{
  nodes.push_back(std::make_shared<CoordinateSystemNode>());
}

std::vector<std::shared_ptr<LabelNode>> Nodes::getLabelNodes()
{
  std::vector<std::shared_ptr<LabelNode>> result;
  for (auto &node : nodes)
  {
    std::shared_ptr<LabelNode> labelNode =
        std::dynamic_pointer_cast<LabelNode>(node);
    if (labelNode.get())
      result.push_back(labelNode);
  }

  return result;
}

void Nodes::addSceneNodesFrom(QUrl url)
{
  addSceneNodesFrom(url.path().toStdString());
}

void Nodes::addSceneNodesFrom(std::string filename)
{
  qDebug() << "Nodes::addSceneNodesFrom" << filename.c_str();
  auto loadedNodes =
      Persister::load<std::vector<std::shared_ptr<Node>>>(filename);

  for (auto &m : loadedNodes)
    nodes.push_back(m);
}

void Nodes::importFrom(std::string filename)
{
  Importer importer;

  auto meshes = importer.importAll(filename);

  for (size_t i = 0; i < meshes.size(); ++i)
  {
    nodes.push_back(
        std::make_shared<MeshNode>(filename, i, meshes[i], Eigen::Matrix4f()));
  }
}

void Nodes::importFrom(QUrl url)
{
  importFrom(url.path().toStdString());
}

void Nodes::render(Gl *gl, RenderData renderData)
{
  for (auto &node : nodes)
    node->render(gl, renderData);

  if (showBoundingVolumes)
  {
    for (auto &node : obbNodes)
      node->render(gl, renderData);
  }
}

void Nodes::saveSceneTo(QUrl url)
{
  saveSceneTo(url.path().toStdString());
}

void Nodes::saveSceneTo(std::string filename)
{
  Persister::save(nodes, filename);
}

void Nodes::clear()
{
  nodes.clear();
}

void Nodes::toggleBoundingVolumes()
{
  showBoundingVolumes = !showBoundingVolumes;

  if (showBoundingVolumes)
  {
    obbNodes.clear();
    for (auto &node : nodes)
    {
      if (node->getObb().get())
      {
        obbNodes.push_back(std::make_shared<ObbNode>(node->getObb()));
      }
    }
  }
}

