#include "./transfer_function_manager.h"
#include <QLoggingCategory>
#include <stdexcept>
#include <map>
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
  if (rowsCache.count(path))
    return rowsCache[path];

  auto vector =
      GradientUtils::loadGradientAsFloats(QString(path.c_str()), width);

  if (textureId < 0)
  {
    textureId = textureManager->addTexture(nullptr, width, height);
  }

  auto texture = textureManager->getTextureFor(textureId);
  texture->texSubImage2D(0, 0, usedRows, width, 1, GL_RGBA, GL_FLOAT,
                         vector.data());
  rowsCache[path] = usedRows;

  return usedRows++;
}

TextureAddress TransferFunctionManager::getTextureAddress()
{
  return textureManager->getAddressFor(textureId);
}

}  // namespace Graphics
