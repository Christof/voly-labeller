#ifndef SRC_COORDINATE_SYSTEM_NODE_H_

#define SRC_COORDINATE_SYSTEM_NODE_H_

#include <memory>
#include "./node.h"
#include "./render_data.h"
#include "./gl.h"
#include "./connector.h"

/**
 * \brief
 *
 *
 */
class CoordinateSystemNode : public Node
{
 public:
  CoordinateSystemNode();
  virtual ~CoordinateSystemNode();

  void render(Gl *gl, RenderData renderData);

 private:
  std::shared_ptr<Connector> x;
  std::shared_ptr<Connector> y;
  std::shared_ptr<Connector> z;
};

#endif  // SRC_COORDINATE_SYSTEM_NODE_H_
