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
class ObjectManager;

/**
 * \brief Base class for easier access to a RenderObject
 */
class Renderable
{
 public:
  Renderable(std::string vertexShaderPath, std::string fragmentShaderPath);
  virtual ~Renderable();

  void initialize(Gl *gl, std::shared_ptr<ObjectManager> objectManager);

  virtual void render(Gl *gl, std::shared_ptr<ObjectManager> objectManager,
              const RenderData &renderData);

 protected:
  virtual void createBuffers(std::shared_ptr<RenderObject> renderObject,
                             std::shared_ptr<ObjectManager> objectManager) = 0;
  virtual void setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                           const RenderData &renderData) = 0;

  std::shared_ptr<RenderObject> renderObject;
  std::shared_ptr<ObjectManager> objectManager;

 private:
  std::string vertexShaderPath;
  std::string fragmentShaderPath;
};

}  // namespace Graphics
#endif  // SRC_GRAPHICS_RENDERABLE_H_
