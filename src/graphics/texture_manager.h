#ifndef SRC_GRAPHICS_TEXTURE_MANAGER_H_

#define SRC_GRAPHICS_TEXTURE_MANAGER_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "./texture_address.h"
#include "./texture_space_description.h"

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

  Texture2d *newTexture2d(TextureSpaceDescription spaceDescription);
  Texture2d *newTexture2d(std::string path);

  void free(Texture2d *texture);

  // maxNumTextures <= 0 will cause allocation of maximum number of layers
  bool initialize(Gl *gl, bool sparse = true, int maxTextureArrayLevels = -1);
  void shutdown();

  TextureAddress getAddressFor(int textureId);

 private:
  Gl *gl;
  std::vector<Texture2d *> textures;
  std::map<TextureSpaceDescription, std::vector<TextureContainer *>>
      textureContainers;
  int maxTextureArrayLevels;
  bool isSparse;

  Texture2d *allocateTexture2d(TextureSpaceDescription spaceDescription);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE_MANAGER_H_
