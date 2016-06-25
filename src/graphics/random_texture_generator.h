#ifndef SRC_GRAPHICS_RANDOM_TEXTURE_GENERATOR_H_

#define SRC_GRAPHICS_RANDOM_TEXTURE_GENERATOR_H_

#include <memory>
#include <random>

namespace Graphics
{

class TextureManager;

/**
 * \brief
 *
 *
 */
class RandomTextureGenerator
{
 public:
  explicit RandomTextureGenerator(
      std::shared_ptr<TextureManager> textureManager);

  int create(int width, int height);

 private:
  std::shared_ptr<TextureManager> textureManager;
  std::default_random_engine gen;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_RANDOM_TEXTURE_GENERATOR_H_
