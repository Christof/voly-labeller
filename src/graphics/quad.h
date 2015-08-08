#ifndef SRC_GRAPHICS_QUAD_H_

#define SRC_GRAPHICS_QUAD_H_

#include <memory>
#include <string>
#include "./render_data.h"
#include "./renderable.h"
#include "./object_manager.h"

namespace Graphics
{

class Gl;
class ShaderProgram;

/**
 * \brief Class to draw a quad which is used for the label
 */
class Quad : public Graphics::Renderable
{
 public:
  Quad();
  Quad(std::string vertexShaderFilename, std::string fragmentShaderFilename);
  virtual ~Quad();

 protected:
  virtual ObjectData createBuffers(std::shared_ptr<ObjectManager> objectManager,
                             std::shared_ptr<TextureManager> textureManager,
                             std::shared_ptr<ShaderManager> shaderManager);

  static const int indexCount = 6;
  static ObjectData staticObjectData;

 private:
  std::string vertexShaderFilename;
  std::string fragmentShaderFilename;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_QUAD_H_
