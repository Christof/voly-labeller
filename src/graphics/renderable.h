#ifndef SRC_GRAPHICS_RENDERABLE_H_

#define SRC_GRAPHICS_RENDERABLE_H_

#include <memory>
#include <string>
#include "./render_data.h"

namespace Graphics
{

class Gl;
class RenderObject;
class ShaderProgram;

/**
 * \brief Base class for easier access to a RenderObject
 */
class Renderable
{
 public:
  Renderable(std::string vertexShaderPath, std::string fragmentShaderPath);
  virtual ~Renderable();

  void initialize(Gl *gl);

  void render(Gl *gl, const RenderData &renderData);

  void setShaderProgram(std::shared_ptr<ShaderProgram> shaderProgram);
  std::shared_ptr<ShaderProgram> getShaderProgram();

 protected:
  virtual void createBuffers(std::shared_ptr<RenderObject> renderObject) = 0;
  virtual void setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                           const RenderData &renderData) = 0;
  virtual void draw(Gl *gl) = 0;

 private:
  std::string vertexShaderPath;
  std::string fragmentShaderPath;
  std::shared_ptr<RenderObject> renderObject;
};

}  // namespace Graphics
#endif  // SRC_GRAPHICS_RENDERABLE_H_
