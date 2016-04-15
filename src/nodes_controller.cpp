#include "./nodes_controller.h"
#include "./nodes.h"

NodesController::NodesController(std::shared_ptr<Nodes> nodes) : nodes(nodes)
{
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

