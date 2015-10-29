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
class Managers;
class ObjectManager;
class TextureManager;
class ShaderManager;

/**
 * \brief Base class for easier access to a RenderObject
 */
class Renderable
{
 public:
  Renderable();
  virtual ~Renderable();

  virtual void initialize(Gl *gl, std::shared_ptr<Managers> managers);

  virtual void render(Gl *gl, std::shared_ptr<Managers> managers,
                      const RenderData &renderData);
  virtual void renderImmediately(Gl *gl, std::shared_ptr<Managers> managers,
                                 const RenderData &renderData);

  ObjectData getObjectData();

 protected:
  virtual ObjectData
  createBuffers(std::shared_ptr<ObjectManager> objectManager,
                std::shared_ptr<TextureManager> textureManager,
                std::shared_ptr<ShaderManager> shaderManager) = 0;

  ObjectData objectData;
};

}  // namespace Graphics
#endif  // SRC_GRAPHICS_RENDERABLE_H_
