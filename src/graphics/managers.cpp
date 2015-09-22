#include "./managers.h"
#include "./object_manager.h"
#include "./texture_manager.h"
#include "./shader_manager.h"

namespace Graphics
{

Managers::Managers()
{
  textureManager = std::make_shared<Graphics::TextureManager>();
  shaderManager = std::make_shared<Graphics::ShaderManager>();
  objectManager =
      std::make_shared<Graphics::ObjectManager>(textureManager, shaderManager);
}

std::shared_ptr<Graphics::ObjectManager> Managers::getObjectManager() const
{
  return objectManager;
}

std::shared_ptr<Graphics::TextureManager> Managers::getTextureManager() const
{
  return textureManager;
}

std::shared_ptr<Graphics::ShaderManager> Managers::getShaderManager() const
{
  return shaderManager;
}

}  // namespace Graphics