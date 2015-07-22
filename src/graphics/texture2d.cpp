#include "./texture2d.h"
#include "./texture_container.h"

namespace Graphics
{

Texture2d::Texture2d(TextureContainer *container, GLsizei sliceCount)
  : container(container), sliceCount(sliceCount)
{
}

Texture2d::~Texture2d()
{
  free();
  container->virtualFree(sliceCount);
}

const TextureContainer *Texture2d::getTextureContainer() const
{
  return container;
}

GLsizei Texture2d::getSliceCount() const
{
  return sliceCount;
}

TextureAddress Texture2d::address() const
{
  printf(" %f %f:\n", static_cast<float>(width) / container->getWidth(),
         static_cast<float>(height) / container->getHeight());
  TextureAddress ta = { container->getHandle(),
                        static_cast<GLfloat>(sliceCount),
                        0,
                        { static_cast<float>(width) / container->getWidth(),
                          static_cast<float>(height) /
                              container->getHeight() } };
  return ta;
}

void Texture2d::commit()
{
  container->commit(this);
}

void Texture2d::free()
{
  container->free(this);
}

void Texture2d::compressedTexSubImage2D(GLint level, GLint xoffset,
                                        GLint yoffset, GLsizei width,
                                        GLsizei height, GLenum format,
                                        GLsizei imageSize, const GLvoid *data)
{
  this->width = width;
  this->height = height;

  container->compressedTexSubImage3d(level, xoffset, yoffset, sliceCount, width,
                                     height, 1, format, imageSize, data);
}

void Texture2d::texSubImage2D(GLint level, GLint xoffset, GLint yoffset,
                              GLsizei width, GLsizei height, GLenum format,
                              GLenum type, const GLvoid *data)
{
  this->width = width;
  this->height = height;

  container->texSubImage3d(level, xoffset, yoffset, sliceCount, width, height,
                           1, format, type, data);
}

}  // namespace Graphics
