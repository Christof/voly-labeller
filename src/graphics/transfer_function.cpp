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

  auto image =
      GradientUtils::loadGradientAsImage(QString(path.c_str()), QSize(512, 1));
  texture = textureManager->addTexture(&image);
}

TransferFunction::~TransferFunction()
{
}

}  // namespace Graphics
