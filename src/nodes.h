#ifndef SRC_NODES_H_

#define SRC_NODES_H_

#include "./node.h"
#include "./render_data.h"
#include <string>
#include <memory>

/**
 * \brief
 *
 *
 */
class Nodes
{
 public:
  Nodes();

  void addSceneNodesFrom(std::string filename);

  void render(RenderData renderData);

 private:
  std::vector<std::shared_ptr<Node>> nodes;
};

#endif  // SRC_NODES_H_
