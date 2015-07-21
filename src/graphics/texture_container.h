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

  void commit(Texture2d *_tex);
  void free(Texture2d *_tex);

  void CompressedTexSubImage3D(GLint level, GLint xoffset, GLint yoffset,
                               GLint zoffset, GLsizei width, GLsizei height,
                               GLsizei depth, GLenum format, GLsizei imageSize,
                               const GLvoid *data);
  void TexSubImage3D(GLint level, GLint xoffset, GLint yoffset, GLint zoffset,
                     GLsizei width, GLsizei height, GLsizei depth,
                     GLenum format, GLenum type, const GLvoid *data);

  GLuint64 getHandle() const;
  GLsizei width() const
  {
    return mWidth;
  }
  GLsizei height() const
  {
    return mHeight;
  }

 private:
  Gl *gl;
  GLuint64 handle = 0;
  GLuint mTexId;
  std::queue<GLsizei> mFreeList;

  const GLsizei mWidth;
  const GLsizei mHeight;
  const GLsizei mLevels;
  const GLsizei mSlices;
  GLsizei mXTileSize = 0;
  GLsizei mYTileSize = 0;

  void changeCommitment(GLsizei slice, GLboolean commit);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE_CONTAINER_H_
