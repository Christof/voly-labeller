#ifndef SRC_CONNECTOR_H_

#define SRC_CONNECTOR_H_

#include <Eigen/Core>
#include <vector>
#include "./render_data.h"
#include "./renderable.h"

class Gl;
class ShaderProgram;

/**
 * \brief Draws a line between two points
 *
 * It is mainly intended to draw the line between an
 * anchor and the corresponding label.
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

 protected:
  virtual void createBuffers(std::shared_ptr<RenderObject> renderObject);
  virtual void setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                           const RenderData &renderData);
  virtual void draw(Gl *gl);

 private:
  std::vector<Eigen::Vector3f> points;
};

#endif  // SRC_CONNECTOR_H_
