#ifndef SRC_GRAPHICS_TEXTURE_MANAGER_H_

#define SRC_GRAPHICS_TEXTURE_MANAGER_H_

#include <QImage>
#include <Eigen/Core>
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
  int addTexture(QImage *image);
  int addTexture(float *data, int width, int height);

  std::shared_ptr<Texture2d>
  newTexture2d(TextureSpaceDescription spaceDescription);
  std::shared_ptr<Texture2d> newTexture2d(std::string path);
  std::shared_ptr<Texture2d> newTexture2d(QImage *image);
  std::shared_ptr<Texture2d> newTexture2d(float *data, int width, int height);

  void free(Texture2d *texture);

  // maxNumTextures <= 0 will cause allocation of maximum number of layers
  bool initialize(Gl *gl, bool sparse = true, int maxTextureArrayLevels = -1);
  void shutdown();

  std::shared_ptr<Texture2d> getTextureFor(int textureId);
  TextureAddress getAddressFor(int textureId);

 private:
  Gl *gl;
  std::vector<std::shared_ptr<Texture2d>> textures;
  std::map<TextureSpaceDescription, std::vector<TextureContainer *>>
      textureContainers;
  int maxTextureArrayLevels;
  bool isSparse;

  std::shared_ptr<Texture2d>
  allocateTexture2d(TextureSpaceDescription spaceDescription);
  int get2DVirtualPageSizeX(int internalFormat);
  int get2DVirtualPageSizeY(int internalFormat);
  int getInternalFormat(int target, int internalFormat, int parameterName);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE_MANAGER_H_
