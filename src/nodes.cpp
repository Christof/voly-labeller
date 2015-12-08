#include "./nodes.h"
#include <QDebug>
#include <string>
#include <vector>
#include "./utils/persister.h"
#include "./importer.h"
#include "./mesh_node.h"
#include "./label_node.h"
#include "./obb_node.h"
#include "./coordinate_system_node.h"

Nodes::Nodes()
{
  addNode(std::make_shared<CoordinateSystemNode>());
}

Nodes::~Nodes()
{
  qInfo() << "Destructor of Nodes";
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

void Nodes::addNode(std::shared_ptr<Node> node)
{
  nodes.push_back(node);

  emit nodesChanged(node);
}

void Nodes::removeNode(std::shared_ptr<Node> node)
{
  nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
}

std::vector<std::shared_ptr<Node>> Nodes::getNodes()
{
  return nodes;
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
    addNode(m);
}

void Nodes::importFrom(std::string filename)
{
  Importer importer;

  auto meshes = importer.importAll(filename);

  for (size_t i = 0; i < meshes.size(); ++i)
  {
    addNode(
        std::make_shared<MeshNode>(filename, i, meshes[i], Eigen::Matrix4f()));
  }
}

void Nodes::importFrom(QUrl url)
{
  importFrom(url.path().toStdString());
}

void Nodes::render(Graphics::Gl *gl,
                   std::shared_ptr<Graphics::Managers> managers,
                   RenderData renderData)
{
  for (auto &node : nodes)
    node->render(gl, managers, renderData);

  if (showBoundingVolumes)
  {
    for (auto &node : obbNodes)
      node->render(gl, managers, renderData);
  }
}

void Nodes::renderLabels(Graphics::Gl *gl,
                   std::shared_ptr<Graphics::Managers> managers,
                   RenderData renderData)
{
  for (auto labelNode : getLabelNodes())
  {
    labelNode->renderLabelAndConnector(gl, managers, renderData);
  }
}

void Nodes::saveSceneTo(QUrl url)
{
  saveSceneTo(url.path().toStdString());
}

void Nodes::saveSceneTo(std::string filename)
{
  std::vector<std::shared_ptr<Node>> persistableNodes;
  for (auto node : nodes)
    if (node->isPersistable())
      persistableNodes.push_back(node);

  Persister::save(persistableNodes, filename);
}

void Nodes::clear()
{
  nodes.clear();
  obbNodes.clear();
}

void Nodes::toggleBoundingVolumes()
{
  showBoundingVolumes = !showBoundingVolumes;

  if (showBoundingVolumes)
  {
    obbNodes.clear();
    for (auto &node : nodes)
    {
      if (node->getObb().isInitialized())
      {
        obbNodes.push_back(std::make_shared<ObbNode>(node->getObb()));
      }
    }
  }
}

void Nodes::addForcesVisualizerNode(std::shared_ptr<Node> node)
{
  addNode(node);
  forcesVisualizerNode = node;
}

void Nodes::removeForcesVisualizerNode()
{
  removeNode(forcesVisualizerNode);
}

