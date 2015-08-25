#include "./transfer_function.h"
#include <QLoggingCategory>
#include <stdexcept>
#include "./texture_manager.h"
#include "../utils/gradient_utils.h"

namespace Graphics
{

// QLoggingCategory tfChan("Graphics.TransferFunction");

TransferFunction::TransferFunction(
    std::shared_ptr<TextureManager> textureManager, std::string path)
  : textureManager(textureManager)
{

  auto vector = GradientUtils::loadGradientAsFloats(QString(path.c_str()), 512);
  texture = textureManager->addTexture(vector.data(), 512, 1);
}

TransferFunction::~TransferFunction()
{
}

}  // namespace Graphics
