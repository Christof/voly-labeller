#ifndef SRC_GRAPHICS_TEXTURE_MANAGER_H_

#define SRC_GRAPHICS_TEXTURE_MANAGER_H_

#include <string>
#include <vector>
#include <memory>
#include "./texture_address.h"

namespace Graphics
{

class TextureContainer;
class Texture2d;
class Gl;

/**
 * \brief
 *
 *
 */
class TextureManager
{
 public:
  TextureManager() = default;
  ~TextureManager();

  int addTexture(std::string path);

  Texture2d *newTexture2d(int levels, int internalFormat, int width,
                          int height);
  Texture2d *newTexture2d(std::string path);

  void free(Texture2d *texture);

  // maxNumTextures <= 0 will cause allocation of maximum number of layers
  bool initialize(Gl *gl, bool sparse = true, int maxTextureArrayLevels = -1);
  void shutdown();

  TextureAddress getAddressFor(int textureId);

 private:
  Gl *gl;
  std::vector<Texture2d *> textures;
  std::map<std::tuple<int, int, int, int>, std::vector<TextureContainer *>>
      textureContainers;
  int maxTextureArrayLevels;
  bool isSparse;

  Texture2d *allocateTexture2d(int levels, int internalformat, int width,
                               int height);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE_MANAGER_H_
