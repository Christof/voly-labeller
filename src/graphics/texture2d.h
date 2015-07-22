#ifndef SRC_GRAPHICS_TEXTURE2D_H_

#define SRC_GRAPHICS_TEXTURE2D_H_

#include "./gl.h"
#include "./texture_address.h"

namespace Graphics
{

class TextureContainer;

/**
 * \brief
 *
 *
 */
class Texture2d
{
 public:
  Texture2d(TextureContainer *container, GLsizei sliceCount);
  ~Texture2d();
  void commit();
  void free();

  void compressedTexSubImage2D(GLint level, GLint xOffset, GLint yOffset,
                               GLsizei width, GLsizei height, GLenum format,
                               GLsizei imageSize, const GLvoid *data);
  void texSubImage2D(GLint level, GLint xOffset, GLint yOffset, GLsizei width,
                     GLsizei height, GLenum format, GLenum type,
                     const GLvoid *data);

  const TextureContainer *getTextureContainer() const;

  GLsizei getSliceCount() const;

  TextureAddress address() const;

 private:
  TextureContainer *container;

  GLsizei width;
  GLsizei height;
  GLsizei sliceCount;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE2D_H_
