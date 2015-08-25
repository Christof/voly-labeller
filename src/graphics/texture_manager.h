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
  int addTexture(float* data, int width, int height);
  unsigned int add3dTexture(Eigen::Vector3i size, float *data);

  Texture2d *newTexture2d(TextureSpaceDescription spaceDescription);
  Texture2d *newTexture2d(std::string path);
  Texture2d *newTexture2d(QImage *image);
  Texture2d *newTexture2d(float* data, int width, int height);

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
  int get2DVirtualPageSizeX(int internalFormat);
  int get2DVirtualPageSizeY(int internalFormat);
  int getInternalFormat(int target, int internalFormat, int parameterName);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE_MANAGER_H_
