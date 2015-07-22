#include "./texture2d.h"
#include "./texture_container.h"

namespace Graphics
{

Texture2d::Texture2d(TextureContainer *container, GLsizei sliceCount)
  : mContainer(container), sliceCount(sliceCount)
{
}

Texture2d::~Texture2d()
{
  free();
  mContainer->virtualFree(sliceCount);
}

const TextureContainer *Texture2d::getTextureContainer() const
{
  return mContainer;
}

GLsizei Texture2d::getSliceCount() const
{
  return sliceCount;
}

TextureAddress Texture2d::address() const
{
  printf(" %f %f:\n", static_cast<float>(mWidth) / mContainer->getWidth(),
         static_cast<float>(mHeight) / mContainer->getHeight());
  TextureAddress ta = { mContainer->getHandle(),
                        static_cast<GLfloat>(sliceCount),
                        0,
                        { static_cast<float>(mWidth) / mContainer->getWidth(),
                          static_cast<float>(mHeight) /
                              mContainer->getHeight() } };
  return ta;
}

void Texture2d::commit()
{
  mContainer->commit(this);
}

void Texture2d::free()
{
  mContainer->free(this);
}

void Texture2d::compressedTexSubImage2D(GLint level, GLint xoffset,
                                        GLint yoffset, GLsizei width,
                                        GLsizei height, GLenum format,
                                        GLsizei imageSize, const GLvoid *data)
{
  mWidth = width;
  mHeight = height;

  mContainer->compressedTexSubImage3d(level, xoffset, yoffset, sliceCount,
                                      width, height, 1, format, imageSize,
                                      data);
}

void Texture2d::texSubImage2D(GLint level, GLint xoffset, GLint yoffset,
                              GLsizei width, GLsizei height, GLenum format,
                              GLenum type, const GLvoid *data)
{
  mWidth = width;
  mHeight = height;
  mContainer->texSubImage3d(level, xoffset, yoffset, sliceCount, width,
                            height, 1, format, type, data);
}

}  // namespace Graphics
