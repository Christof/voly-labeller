#ifndef SRC_GRAPHICS_CONNECTOR_H_

#define SRC_GRAPHICS_CONNECTOR_H_

#include <Eigen/Core>
#include <vector>
#include <string>
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
  Connector(std::string vertexShaderFilename,
            std::string fragmentShaderFilename, Eigen::Vector3f anchor,
            Eigen::Vector3f label);
  explicit Connector(
      std::vector<Eigen::Vector3f> points,
      std::string vertexShaderFilename = ":/shader/pass.vert",
      std::string fragmentShaderFilename = ":/shader/color.frag");

  Eigen::Vector4f color;
  float lineWidth = 3.0f;
  float zOffset = 0.0f;

 protected:
  virtual ObjectData
  createBuffers(std::shared_ptr<ObjectManager> objectManager,
                std::shared_ptr<TextureManager> textureManager,
                std::shared_ptr<ShaderManager> shaderManager);

 private:
  std::vector<Eigen::Vector3f> points;
  std::string vertexShaderFilename;
  std::string fragmentShaderFilename;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_CONNECTOR_H_
