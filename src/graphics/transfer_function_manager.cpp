#include "./transfer_function_manager.h"
#include <QLoggingCategory>
#include <stdexcept>
#include "./texture_manager.h"
#include "./texture2d.h"
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

  if (textureId < 0)
  {
    textureId = textureManager->addTexture(vector.data(), width, 64);
    return usedRows++;
  }

  auto texture = textureManager->getTextureFor(textureId);
  texture->texSubImage2D(0, 0, ++usedRows, width, 1, GL_RGBA, GL_FLOAT,
                         vector.data());

  return usedRows;
}

}  // namespace Graphics
