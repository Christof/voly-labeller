#ifndef SRC_GRAPHICS_TEXTURE_CONTAINER_H_

#define SRC_GRAPHICS_TEXTURE_CONTAINER_H_

#include <queue>
#include "./gl.h"
#include "./texture_space_description.h"

namespace Graphics
{
class Texture2d;

class TextureContainer
{
 public:
  TextureContainer(Gl *gl, bool sparse,
                   TextureSpaceDescription spaceDescription, int slices);
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

  const TextureSpaceDescription spaceDescription;
  const int slices;

  void changeCommitment(GLsizei slice, GLboolean commit);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE_CONTAINER_H_
