#ifndef SRC_COORDINATE_SYSTEM_NODE_H_

#define SRC_COORDINATE_SYSTEM_NODE_H_

#include <memory>
#include "./node.h"
#include "./render_data.h"
#include "./graphics/connector.h"
#include "./graphics/gl.h"

/**
 * \brief Node which draws a coordinate system
 *
 * Red is the x-axis, green the y-axis and blue
 * the z-axis.
 */
class CoordinateSystemNode : public Node
{
 public:
  CoordinateSystemNode();
  virtual ~CoordinateSystemNode();

  void render(Graphics::Gl *gl, RenderData renderData);

 private:
  std::shared_ptr<Graphics::Connector> x;
  std::shared_ptr<Graphics::Connector> y;
  std::shared_ptr<Graphics::Connector> z;
};

#endif  // SRC_COORDINATE_SYSTEM_NODE_H_
