#ifndef SRC_GRAPHICS_CUBE_H_

#define SRC_GRAPHICS_CUBE_H_

#include <Eigen/Core>
#include <vector>
#include <string>
#include "./render_data.h"
#include "./renderable.h"
#include "./object_manager.h"

namespace Graphics
{

/**
 * \brief Draws a cube of size 1x1x1 centered at the origin
 *
 */
class Cube : public Renderable
{
 public:
  Cube(std::string vertexShaderPath, std::string fragmentShaderPath);

 protected:
  virtual void createBuffers(std::shared_ptr<RenderObject> renderObject,
                             std::shared_ptr<ObjectManager> objectManager);
  virtual void setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                           const RenderData &renderData);

 private:
  std::vector<Eigen::Vector3f> points;
  static const int indexCount = 36;
  ObjectData objectData;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_CUBE_H_
