#ifndef SRC_OBB_NODE_H_

#define SRC_OBB_NODE_H_

#include <memory>
#include "./node.h"
#include "./math/obb.h"
#include "./graphics/connector.h"
#include "./graphics/gl.h"

/**
 * \brief Node which displays an Obb
 */
class ObbNode : public Node
{
 public:
  explicit ObbNode(const Math::Obb &obb);

  virtual void render(Graphics::Gl *gl,
                      std::shared_ptr<Graphics::Managers> managers,
                      RenderData renderData);

 private:
  std::shared_ptr<Graphics::Connector> wireframe;
};

#endif  // SRC_OBB_NODE_H_
