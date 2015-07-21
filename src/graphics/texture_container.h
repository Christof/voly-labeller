#ifndef SRC_GRAPHICS_TEXTURE_CONTAINER_H_

#define SRC_GRAPHICS_TEXTURE_CONTAINER_H_

#include <queue>
#include "./gl.h"

namespace Graphics
{

class Texture2d;

class TextureContainer
{
 public:
  TextureContainer(Gl *gl, bool sparse, GLsizei levels, GLenum internalformat,
                   GLsizei width, GLsizei height, GLsizei slices);
  ~TextureContainer();
  GLsizei hasRoom() const;
  GLsizei virtualAlloc();
  void virtualFree(GLsizei slice);

  void commit(Texture2d *texture);
  void free(Texture2d *texture);

  void compressedTexSubImage3d(GLint level, GLint xoffset, GLint yoffset,
                               GLint zoffset, GLsizei width, GLsizei height,
                               GLsizei depth, GLenum format, GLsizei imageSize,
                               const GLvoid *data);
  void texSubImage3d(GLint level, GLint xoffset, GLint yoffset, GLint zoffset,
                     GLsizei width, GLsizei height, GLsizei depth,
                     GLenum format, GLenum type, const GLvoid *data);

  GLuint64 getHandle() const;
  GLsizei getWidth() const;
  GLsizei getHeight() const;

 private:
  Gl *gl;
  GLuint64 handle = 0;
  GLuint textureId;
  std::queue<GLsizei> freeList;

  const GLsizei width;
  const GLsizei height;
  const GLsizei levels;
  const GLsizei slices;

  void changeCommitment(GLsizei slice, GLboolean commit);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE_CONTAINER_H_
