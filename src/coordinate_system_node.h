#ifndef SRC_COORDINATE_SYSTEM_NODE_H_

#define SRC_COORDINATE_SYSTEM_NODE_H_

#include <memory>
#include "./node.h"
#include "./graphics/render_data.h"
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

  virtual void render(Graphics::Gl *gl,
                      std::shared_ptr<Graphics::Managers> managers,
                      RenderData renderData);

 private:
  std::shared_ptr<Graphics::Connector> x;
  std::shared_ptr<Graphics::Connector> y;
  std::shared_ptr<Graphics::Connector> z;
};

#endif  // SRC_COORDINATE_SYSTEM_NODE_H_
