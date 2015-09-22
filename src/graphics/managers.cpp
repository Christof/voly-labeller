#include "./managers.h"
#include "./object_manager.h"
#include "./texture_manager.h"
#include "./shader_manager.h"
#include "./transfer_function_manager.h"
#include "./volume_manager.h"

namespace Graphics
{

Managers::Managers()
{
  textureManager = std::make_shared<TextureManager>();
  shaderManager = std::make_shared<ShaderManager>();
  objectManager =
      std::make_shared<ObjectManager>(textureManager, shaderManager);
  transferFunctionManager =
      std::make_shared<TransferFunctionManager>(textureManager);
  volumeManager = std::make_shared<VolumeManager>();
}

std::shared_ptr<ObjectManager> Managers::getObjectManager() const
{
  return objectManager;
}

std::shared_ptr<TextureManager> Managers::getTextureManager() const
{
  return textureManager;
}

std::shared_ptr<ShaderManager> Managers::getShaderManager() const
{
  return shaderManager;
}

std::shared_ptr<TransferFunctionManager>
Managers::getTransferFunctionManager() const
{
  return transferFunctionManager;
}

std::shared_ptr<VolumeManager> Managers::getVolumeManager() const
{
  return volumeManager;
}

}  // namespace Graphics
