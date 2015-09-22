#ifndef SRC_GRAPHICS_MANAGERS_H_

#define SRC_GRAPHICS_MANAGERS_H_

#include <memory>

namespace Graphics
{

class ObjectManager;
class TextureManager;
class ShaderManager;
class TransferFunctionManager;

/**
 * \brief
 *
 *
 */
class Managers
{
 public:
  Managers();

  std::shared_ptr<ObjectManager> getObjectManager() const;
  std::shared_ptr<TextureManager> getTextureManager() const;
  std::shared_ptr<ShaderManager> getShaderManager() const;
  std::shared_ptr<TransferFunctionManager> getTransferFunctionManager() const;

 private:
  std::shared_ptr<ObjectManager> objectManager;
  std::shared_ptr<TextureManager> textureManager;
  std::shared_ptr<ShaderManager> shaderManager;
  std::shared_ptr<TransferFunctionManager> transferFunctionManager;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_MANAGERS_H_
