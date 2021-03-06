#include "./texture_mapper_manager_controller.h"
#include <iostream>

TextureMapperManagerController::TextureMapperManagerController(
    std::shared_ptr<TextureMapperManager> textureMapperManager)
  : textureMapperManager(textureMapperManager)
{
}

void TextureMapperManagerController::saveDistanceTransform()
{
  textureMapperManager->saveDistanceTransform();
}

void TextureMapperManagerController::saveApollonius()
{
  textureMapperManager->saveApollonius();
}

void TextureMapperManagerController::saveSaliency()
{
  textureMapperManager->saveSaliency();
}
