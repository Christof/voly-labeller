#include "./random_texture_generator.h"
#include <vector>
#include "./texture_manager.h"

namespace Graphics
{

RandomTextureGenerator::RandomTextureGenerator(
    std::shared_ptr<TextureManager> textureManager)
  : textureManager(textureManager)
{
}

int RandomTextureGenerator::create(int width, int height)
{
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  int pixelCount = width * height * 4;
  std::vector<float> data(pixelCount);

  for (int index = 0; index < pixelCount; ++index)
    data[index] = dist(gen);

  return textureManager->addTexture(data.data(), width, height);
}

}  // namespace Graphics
