#ifndef SRC_GRAPHICS_TEXTURE_MANAGER_H_

#define SRC_GRAPHICS_TEXTURE_MANAGER_H_

#include <string>
#include <vector>
#include <memory>
#include "./texture_address.h"
#include "./gl.h"

namespace Graphics
{

class TextureContainer;
class Texture2d;

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

  Texture2d *newTexture2d(GLsizei levels, GLenum internalformat, GLsizei width,
                          GLsizei height);
  Texture2d *newTexture2d(std::string path);

  void free(Texture2d *_tex);

  // maxNumTextures <= 0 will cause allocation of maximum number of layers
  bool initialize(Gl *gl, bool sparse = true, GLsizei maxNumTextures = -1);
  void shutdown();

  TextureAddress getAddressFor(int textureId);

 private:
  Gl *gl;
  std::vector<Texture2d *> textures;
  std::map<std::tuple<GLsizei, GLenum, GLsizei, GLsizei>,
           std::vector<TextureContainer *>> mTexArrays2D;
  bool mInited;
  GLsizei mMaxTextureArrayLevels;

  bool mSparse;

  Texture2d *allocateTexture2d(GLsizei levels, GLenum internalformat,
                               GLsizei width, GLsizei height);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE_MANAGER_H_
