#ifndef SRC_GRAPHICS_CUBE_H_

#define SRC_GRAPHICS_CUBE_H_

#include <Eigen/Core>
#include <vector>
#include <string>
#include "./render_data.h"
#include "./renderable.h"
#include "./object_manager.h"
#include "./shader_program.h"

namespace Graphics
{

/**
 * \brief Draws a cube of size 1x1x1 centered at the origin
 *
 */
class Cube : public Renderable
{
 public:
  Cube();

 protected:
  virtual ObjectData
  createBuffers(std::shared_ptr<ObjectManager> objectManager,
                std::shared_ptr<TextureManager> textureManager,
                std::shared_ptr<ShaderManager> shaderManager);

 private:
  std::vector<Eigen::Vector3f> points;
  static const int indexCount = 36;
  ObjectData objectData;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_CUBE_H_
