#include "./nodes.h"
#include <QDebug>
#include <string>
#include <vector>
#include "./math/eigen.h"
#include "./utils/persister.h"
#include "./importer.h"
#include "./mesh_node.h"
#include "./volume_node.h"
#include "./label_node.h"
#include "./obb_node.h"
#include "./camera_node.h"
#include "./coordinate_system_node.h"

Nodes::Nodes()
{
  coordinateSystemNode = std::make_shared<CoordinateSystemNode>();
  coordinateSystemNode->setIsVisible(false);
  addNode(coordinateSystemNode);

  cameraNode = std::make_shared<CameraNode>();
  addNode(cameraNode);
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

  if (onNodeAdded)
    onNodeAdded(node);
}

void Nodes::removeNode(std::shared_ptr<Node> node)
{
  nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
}

std::vector<std::shared_ptr<Node>> Nodes::getNodes()
{
  return nodes;
}

std::vector<std::shared_ptr<Node>> Nodes::getNodesForObb()
{
  auto result = getNodes();
  if (cameraOriginVisualizerNode.get())
    result.push_back(cameraOriginVisualizerNode);

  return result;
}

std::shared_ptr<CameraNode> Nodes::getCameraNode()
{
  return cameraNode;
}

void Nodes::setCameraNode(std::shared_ptr<CameraNode> node)
{
  if (cameraNode.get())
    removeNode(cameraNode);

  cameraNode = node;
  addNode(node);
}

void Nodes::addSceneNodesFrom(std::string filename)
{
  qDebug() << "Nodes::addSceneNodesFrom" << filename.c_str();
  auto loadedNodes =
      Persister::load<std::vector<std::shared_ptr<Node>>>(filename);

  for (auto &node : loadedNodes)
  {
    std::shared_ptr<CameraNode> camera =
        std::dynamic_pointer_cast<CameraNode>(node);
    if (camera.get())
      setCameraNode(camera);

    addNode(node);
  }
}

void Nodes::importMeshFrom(std::string filename)
{
  Importer importer;

  auto meshes = importer.importAll(filename);

  for (size_t i = 0; i < meshes.size(); ++i)
  {
    auto transformation =
        importer.getTransformationFor(filename, static_cast<int>(i));
    addNode(std::make_shared<MeshNode>(filename, i, meshes[i], transformation));
  }
}

void Nodes::importVolume(std::string volumeFilename,
                         std::string transferFunctionFilename)
{
  addNode(std::make_shared<VolumeNode>(volumeFilename, transferFunctionFilename,
                                       Eigen::Matrix4f::Identity()));
}

void Nodes::render(Graphics::Gl *gl,
                   std::shared_ptr<Graphics::Managers> managers,
                   RenderData renderData)
{
  auto allNodes = nodes;
  for (auto &node : allNodes)
    node->render(gl, managers, renderData);

  renderCameraOriginVisualizer(gl, managers, renderData);

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

void Nodes::renderOverlays(Graphics::Gl *gl,
                           std::shared_ptr<Graphics::Managers> managers,
                           RenderData renderData)
{
  if (forcesVisualizerNode.get())
  {
    forcesVisualizerNode->render(gl, managers, renderData);
  }
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
  clearForShutdown();

  addNode(cameraNode);
}

void Nodes::clearForShutdown()
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

void Nodes::toggleCoordinateSystem()
{
  coordinateSystemNode->toggleVisibility();
}

void Nodes::toggleCameraOriginVisualizer()
{
  cameraOriginVisualizerNode->toggleVisibility();
}

void Nodes::addForcesVisualizerNode(std::shared_ptr<Node> node)
{
  forcesVisualizerNode = node;
}

void Nodes::removeForcesVisualizerNode()
{
  forcesVisualizerNode.reset();
}

void Nodes::applyTransformationToAllNodes(Eigen::Matrix4f transformation)
{
  for (auto &node : nodes)
  {
    std::shared_ptr<VolumeNode> volumeNode =
        std::dynamic_pointer_cast<VolumeNode>(node);
    if (volumeNode.get())
    {
      volumeNode->setTransformation(transformation *
                                    volumeNode->getTransformation());
    }

    std::shared_ptr<MeshNode> meshNode =
        std::dynamic_pointer_cast<MeshNode>(node);
    if (meshNode.get())
    {
      meshNode->setTransformation(transformation *
                                  meshNode->getTransformation());
    }

    std::shared_ptr<LabelNode> labelNode =
        std::dynamic_pointer_cast<LabelNode>(node);
    if (labelNode.get())
    {
      labelNode->label.anchorPosition =
          mul(transformation, labelNode->label.anchorPosition).head<3>();
    }
  }
}

void Nodes::setOnNodeAdded(
    std::function<void(std::shared_ptr<Node>)> onNodeAdded)
{
  this->onNodeAdded = onNodeAdded;
}

void Nodes::createCameraOriginVisualizer()
{
  Importer importer;

  std::string filename = "assets/models/cameraOrigin.dae";
  auto cameraOriginSphere = importer.import(filename, 0);

  cameraOriginVisualizerNode = std::make_shared<MeshNode>(
      filename, 0, cameraOriginSphere, Eigen::Matrix4f::Identity());
  cameraOriginVisualizerNode->setIsVisible(false);
}

void Nodes::renderCameraOriginVisualizer(
    Graphics::Gl *gl, std::shared_ptr<Graphics::Managers> managers,
    const RenderData &renderData)
{
  if (!cameraOriginVisualizerNode.get())
  {
    createCameraOriginVisualizer();
  }

  auto origin = cameraNode->getCamera()->getOrigin();
  auto originNDC = project(renderData.viewProjectionMatrix, origin);

  float sizeNDC = 16.0f / renderData.windowPixelSize.x();

  Eigen::Vector3f sizeWorld =
      calculateWorldScale(Eigen::Vector4f(sizeNDC, sizeNDC, originNDC.z(), 1),
                          renderData.projectionMatrix);

  Eigen::Affine3f transformation(Eigen::Translation3f(origin) *
                                 Eigen::Scaling(sizeWorld.x()));
  cameraOriginVisualizerNode->setTransformation(transformation.matrix());
  cameraOriginVisualizerNode->render(gl, managers, renderData);
}

