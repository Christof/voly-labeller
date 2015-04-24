#ifndef SRC_OBB_NODE_H_

#define SRC_OBB_NODE_H_

#include <memory>
#include "./node.h"

class Gl;
class Obb;
class Connector;

class ObbNode : public Node
{
 public:
  explicit ObbNode(std::shared_ptr<Obb> obb);

  void render(Gl *gl, RenderData renderData);

 private:
  std::shared_ptr<Connector> wireframe;
};

#endif  // SRC_OBB_NODE_H_
