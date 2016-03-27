#ifndef SRC_GRAPHICS_BUFFER_DRAWER_H_

#define SRC_GRAPHICS_BUFFER_DRAWER_H_

#include <Eigen/Core>
#include <memory>
#include <vector>
#include "./drawer.h"

namespace Graphics
{

class Gl;
class ShaderManager;

/**
 * \brief Drawer implementation which uses OpenGL to draw into the currently
 * bound frame buffer
 *
 */
class BufferDrawer : public Drawer
{
 public:
  BufferDrawer(int width, int height, Gl *gl,
               std::shared_ptr<ShaderManager> shaderManager);

  void drawElementVector(std::vector<float> positions, float color);
  void clear();

 private:
  int width;
  int height;
  Gl *gl;
  std::shared_ptr<Graphics::ShaderManager> shaderManager;
  int shaderId;
  Eigen::Matrix4f pixelToNDC;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_BUFFER_DRAWER_H_
