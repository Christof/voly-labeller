#ifndef SRC_GRAPHICS_QUAD_H_

#define SRC_GRAPHICS_QUAD_H_

#include <memory>
#include <string>
#include "./render_data.h"
#include "./renderable.h"

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
  virtual void
  createBuffers(std::shared_ptr<Graphics::RenderObject> renderObject);
  virtual void
  setUniforms(std::shared_ptr<Graphics::ShaderProgram> shaderProgram,
              const RenderData &renderData);
  virtual void draw(Gl *gl);

 private:
  static const int indexCount = 6;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_QUAD_H_
