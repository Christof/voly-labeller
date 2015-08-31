#include "./transfer_function_manager.h"
#include <QLoggingCategory>
#include <stdexcept>
#include "./texture_manager.h"
#include "../utils/gradient_utils.h"

namespace Graphics
{

// QLoggingCategory tfChan("Graphics.TransferFunction");

TransferFunctionManager::TransferFunctionManager(
    std::shared_ptr<TextureManager> textureManager)
  : textureManager(textureManager)
{
}

TransferFunctionManager::~TransferFunctionManager()
{
}

int TransferFunctionManager::add(std::string path)
{
  auto vector =
      GradientUtils::loadGradientAsFloats(QString(path.c_str()), width);

  if (texture < 0)
  {
    texture = textureManager->addTexture(vector.data(), width, 64);
    return usedRows++;
  }

  throw std::runtime_error("More than one transfer function not yet supported");
}

}  // namespace Graphics
