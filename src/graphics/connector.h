#ifndef SRC_GRAPHICS_CONNECTOR_H_

#define SRC_GRAPHICS_CONNECTOR_H_

#include <Eigen/Core>
#include <vector>
#include "./render_data.h"
#include "./renderable.h"
#include "./object_manager.h"

namespace Graphics
{

class ShaderProgram;
class Gl;

/**
 * \brief Draws a line between two points or many points
 *
 * It is mainly intended to draw the line between an
 * anchor and the corresponding label.
 *
 * It is also used to visualize the Obb.
 */
class Connector : public Renderable
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Connector(Eigen::Vector3f anchor, Eigen::Vector3f label);
  explicit Connector(std::vector<Eigen::Vector3f> points);
  virtual ~Connector();

  Eigen::Vector4f color;
  float lineWidth = 3.0f;
  float zOffset = 0.0f;

 protected:
  virtual void createBuffers(std::shared_ptr<RenderObject> renderObject,
                             std::shared_ptr<ObjectManager> objectManager);
  virtual void setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                           const RenderData &renderData);

 private:
  std::vector<Eigen::Vector3f> points;
  ObjectData objectData;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_CONNECTOR_H_
