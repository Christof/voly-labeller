#include "./nodes_controller.h"
#include "./nodes.h"
#include "./label_node.h"

NodesController::NodesController(std::shared_ptr<Nodes> nodes) : nodes(nodes)
{
}

float NodesController::getAnchorSize()
{
  return labelAnchorSize;
}

void NodesController::addSceneNodesFrom(QUrl url)
{
  nodes->addSceneNodesFrom(url.path().toStdString());
}

void NodesController::importMeshFrom(QUrl url)
{
  nodes->importMeshFrom(url.path().toStdString());
}

void NodesController::setVolumeToImport(QUrl url)
{
  volumeToImport = url.path().toStdString();
}

void NodesController::importVolume(QUrl transferFunctionUrl)
{
  nodes->importVolume(volumeToImport, transferFunctionUrl.path().toStdString());
}

void NodesController::saveSceneTo(QUrl url)
{
  nodes->saveSceneTo(url.path().toStdString());
}

void NodesController::clear()
{
  nodes->clear();
}

void NodesController::toggleBoundingVolumes()
{
  nodes->toggleBoundingVolumes();
}

void NodesController::toggleCoordinateSystem()
{
  nodes->toggleCoordinateSystem();
}

void NodesController::toggleCameraOriginVisualizer()
{
  nodes->toggleCameraOriginVisualizer();
}

void NodesController::changeAnchorSize(float size)
{
  labelAnchorSize = size;

  auto labels = nodes->getLabelNodes();
  for (auto label : labels)
    label->anchorSize = size;
}

