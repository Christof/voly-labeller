#include "./nodes.h"
#include <QDebug>
#include "./utils/persister.h"

Nodes::Nodes()
{
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

void Nodes::render(RenderData renderData)
{
  for (auto &node : nodes)
    node->render(renderData);
}
