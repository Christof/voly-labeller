#ifndef SRC_GRAPHICS_RENDERABLE_H_

#define SRC_GRAPHICS_RENDERABLE_H_

#include <memory>
#include <string>
#include "./render_data.h"
#include "./object_data.h"

namespace Graphics
{

class Gl;
class RenderObject;
class ShaderManager;
class TextureManager;
class ObjectManager;
class ShaderProgram;

/**
 * \brief Base class for easier access to a RenderObject
 */
class Renderable
{
 public:
  Renderable();
  virtual ~Renderable();

  void initialize(Gl *gl, std::shared_ptr<ObjectManager> objectManager,
                  std::shared_ptr<TextureManager> textureManager,
                  std::shared_ptr<ShaderManager> shaderManager);

  virtual void render(Gl *gl, std::shared_ptr<ObjectManager> objectManager,
                      std::shared_ptr<TextureManager> textureManager,
                      std::shared_ptr<ShaderManager> shaderManager,
                      const RenderData &renderData);

 protected:
  virtual ObjectData
  createBuffers(std::shared_ptr<ObjectManager> objectManager,
                std::shared_ptr<TextureManager> textureManager,
                std::shared_ptr<ShaderManager> shaderManager) = 0;

  std::shared_ptr<ObjectManager> objectManager;

  ObjectData objectData;
};

}  // namespace Graphics
#endif  // SRC_GRAPHICS_RENDERABLE_H_
