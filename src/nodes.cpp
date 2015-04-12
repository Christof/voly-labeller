#include "./nodes.h"
#include "./utils/persister.h"

Nodes::Nodes()
{
}

void Nodes::addSceneNodesFrom(std::string filename)
{
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
