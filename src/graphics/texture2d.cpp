#include "./texture2d.h"
#include "./texture_container.h"

namespace Graphics
{

Texture2d::Texture2d(TextureContainer *container, GLsizei sliceNum)
  : mContainer(container), mSliceNum(GLfloat(sliceNum))
{
}

Texture2d::~Texture2d()
{
  free();
  mContainer->virtualFree(GLsizei(mSliceNum));
}

const TextureContainer *Texture2d::getTextureContainer() const
{
  return mContainer;
}

GLsizei Texture2d::getSliceNum() const
{
  return (GLsizei)mSliceNum;
}

TextureAddress Texture2d::address() const
{
  printf(" %f %f:\n", static_cast<float>(mWidth) / mContainer->width(),
         static_cast<float>(mHeight) / mContainer->height());
  TextureAddress ta = { mContainer->getHandle(),
                        mSliceNum,
                        0,
                        { static_cast<float>(mWidth) / mContainer->width(),
                          static_cast<float>(mHeight) /
                              mContainer->height() } };
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

  mContainer->CompressedTexSubImage3D(level, xoffset, yoffset, getSliceNum(),
                                      width, height, 1, format, imageSize,
                                      data);
}

void Texture2d::texSubImage2D(GLint level, GLint xoffset, GLint yoffset,
                              GLsizei width, GLsizei height, GLenum format,
                              GLenum type, const GLvoid *data)
{
  mWidth = width;
  mHeight = height;
  mContainer->TexSubImage3D(level, xoffset, yoffset, getSliceNum(), width,
                            height, 1, format, type, data);
}

}  // namespace Graphics
