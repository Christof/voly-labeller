#ifndef SRC_GRAPHICS_MANAGERS_H_

#define SRC_GRAPHICS_MANAGERS_H_

#include <memory>

namespace Graphics
{

class ObjectManager;
class TextureManager;
class ShaderManager;

/**
 * \brief
 *
 *
 */
class Managers
{
 public:
  Managers();

  std::shared_ptr<Graphics::ObjectManager> getObjectManager() const;
  std::shared_ptr<Graphics::TextureManager> getTextureManager() const;
  std::shared_ptr<Graphics::ShaderManager> getShaderManager() const;

 private:
  std::shared_ptr<Graphics::ObjectManager> objectManager;
  std::shared_ptr<Graphics::TextureManager> textureManager;
  std::shared_ptr<Graphics::ShaderManager> shaderManager;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_MANAGERS_H_
