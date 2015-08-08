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
  virtual void createBuffers(std::shared_ptr<RenderObject> renderObject,
                             std::shared_ptr<ObjectManager> objectManager);
  virtual void setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                           const RenderData &renderData);

  static const int indexCount = 6;
  static ObjectData objectData;
 private:
  std::string vertexShaderFilename;
  std::string fragmentShaderFilename;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_QUAD_H_
