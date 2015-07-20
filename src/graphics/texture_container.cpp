#include "./texture_container.h"
#include <cassert>
#include <iostream>
#include "./texture2d.h"

namespace Graphics
{

TextureContainer::TextureContainer(Gl *gl, bool sparse, GLsizei levels,
                                   GLenum internalformat, GLsizei width,
                                   GLsizei height, GLsizei slices)
  : gl(gl), mWidth(width), mHeight(height), mLevels(levels), mSlices(slices)
{
  glAssert(gl->glGenTextures(1, &mTexId));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D_ARRAY, mTexId));

  if (sparse)
  {
    glAssert(gl->glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_SPARSE_ARB,
                                 GL_TRUE));

    // TODO: This could be done once per internal format. For now, just do it
    // every time.
    GLint indexCount = 0, xSize = 0, ySize = 0, zSize = 0;

    GLint bestIndex = -1, bestXSize = 0, bestYSize = 0;

    glAssert(gl->glGetInternalformativ(GL_TEXTURE_2D_ARRAY, internalformat,
                                       GL_NUM_VIRTUAL_PAGE_SIZES_ARB, 1,
                                       &indexCount));
    std::cout << "Container: indexCount: " << indexCount << " width: " << width
              << " height: " << height << " internal Format: " << internalformat
              << std::endl;

    for (GLint i = 0; i < indexCount; ++i)
    {
      glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_VIRTUAL_PAGE_SIZE_INDEX_ARB, i);
      glAssert(gl->glGetInternalformativ(GL_TEXTURE_2D_ARRAY, internalformat,
                                         GL_VIRTUAL_PAGE_SIZE_X_ARB, 1,
                                         &xSize));
      glAssert(gl->glGetInternalformativ(GL_TEXTURE_2D_ARRAY, internalformat,
                                         GL_VIRTUAL_PAGE_SIZE_Y_ARB, 1,
                                         &ySize));
      glAssert(gl->glGetInternalformativ(GL_TEXTURE_2D_ARRAY, internalformat,
                                         GL_VIRTUAL_PAGE_SIZE_Z_ARB, 1,
                                         &zSize));

      // For our purposes, the "best" format is the one that winds up with Z=1
      // and the largest x and y sizes.
      if (zSize == 1)
      {
        if (xSize >= bestXSize && ySize >= bestYSize)
        {
          bestIndex = i;
          bestXSize = xSize;
          bestYSize = ySize;
        }
      }
    }

    // This would mean the implementation has no valid sizes for us, or that
    // this format doesn't actually support sparse
    // texture allocation. Need to implement the fallback. TODO: Implement that.
    assert(bestIndex != -1);

    mXTileSize = bestXSize;

    glAssert(gl->glTexParameteri(GL_TEXTURE_2D_ARRAY,
                                 GL_VIRTUAL_PAGE_SIZE_INDEX_ARB, bestIndex));
  }

  // We've set all the necessary parameters, now it's time to create the sparse
  // texture.

  std::cout << "Container: levels:" << levels << " width: " << width
            << " height: " << height << " slices: " << mSlices
            << " internal format: " << internalformat << std::endl;

  glAssert(gl->glTexStorage3D(GL_TEXTURE_2D_ARRAY, levels, internalformat,
                              width, height, mSlices));

  const uint tsize = width * height * mSlices * 3;
  unsigned char *tdata = new unsigned char[tsize];
  for (uint i = 0; i < tsize; i++)
    tdata[i] = 0;

  glAssert(gl->glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, width, height,
                               mSlices, GL_RGB, GL_UNSIGNED_BYTE, tdata));

  for (GLsizei i = 0; i < mSlices; ++i)
  {
    mFreeList.push(i);
  }

  if (sparse)
  {
    mHandle = gl->getBindlessTexture()->glGetTextureHandleNV(mTexId);
    glCheckError();
    std::cout << "Container: mTexId: " << mTexId << " handle: " << mHandle
              << std::endl;
    assert(mHandle != 0);
    glAssert(gl->getBindlessTexture()->glMakeTextureHandleResidentNV(mHandle));
  }
}

TextureContainer::~TextureContainer()
{
  // If this fires, it means there was a texture leaked somewhere.
  assert(mFreeList.size() == static_cast<size_t>(mSlices));

  if (mHandle != 0)
  {
    glAssert(
        gl->getBindlessTexture()->glMakeTextureHandleNonResidentNV(mHandle));
    mHandle = 0;
  }
  glAssert(gl->glDeleteTextures(1, &mTexId));
}

GLsizei TextureContainer::hasRoom() const
{
  return mFreeList.size() > 0;
}

GLsizei TextureContainer::virtualAlloc()
{
  GLsizei returnValue = mFreeList.front();
  mFreeList.pop();
  return returnValue;
}

void TextureContainer::virtualFree(GLsizei slice)
{
  mFreeList.push(slice);
}

void TextureContainer::commit(Texture2d *_tex)
{
  assert(_tex->getTextureContainer() == this);

  changeCommitment(_tex->getSliceNum(), GL_TRUE);
}

void TextureContainer::free(Texture2d *_tex)
{
  assert(_tex->getTextureContainer() == this);
  changeCommitment(_tex->getSliceNum(), GL_FALSE);
}

void TextureContainer::CompressedTexSubImage3D(GLint level, GLint xoffset,
                                               GLint yoffset, GLint zoffset,
                                               GLsizei width, GLsizei height,
                                               GLsizei depth, GLenum format,
                                               GLsizei imageSize,
                                               const GLvoid *data)
{
  glAssert(gl->glBindTexture(GL_TEXTURE_2D_ARRAY, mTexId));
  glAssert(gl->glCompressedTexSubImage3D(GL_TEXTURE_2D_ARRAY, level, xoffset,
                                         yoffset, zoffset, width, height, depth,
                                         format, imageSize, data));
}

void TextureContainer::TexSubImage3D(GLint level, GLint xoffset, GLint yoffset,
                                     GLint zoffset, GLsizei width,
                                     GLsizei height, GLsizei depth,
                                     GLenum format, GLenum type,
                                     const GLvoid *data)
{
  glAssert(gl->glBindTexture(GL_TEXTURE_2D_ARRAY, mTexId));

  std::cout << "level: " << level << " xoffset: " << xoffset
            << " yoffset: " << yoffset << " zoffset: " << zoffset
            << " width: " << width << " height: " << height
            << " depth: " << depth << " format: " << format << " type: " << type
            << std::endl;

  /*
  unsigned char *xdata = new unsigned char[width*height*3];
  for (int i =0 ;i< width *height; i++)
  {
    xdata[i*3] = 0.5;
    xdata[i*3 + 1] = 0.1;
     xdata[i*3 + 2] = 1.0;
  }
  glTexSubImage3D(GL_TEXTURE_2D_ARRAY, level, xoffset, yoffset, zoffset,
  width, height, depth, format, type, xdata);

  delete [] xdata;
  */
  glAssert(gl->glTexSubImage3D(GL_TEXTURE_2D_ARRAY, level, xoffset, yoffset,
                               zoffset, width, height, depth, format, type,
                               data));
}

void TextureContainer::changeCommitment(GLsizei slice, GLboolean commit)
{
  GLsizei levelWidth = mWidth;
  GLsizei levelHeight = mHeight;

  for (int level = 0; level < mLevels; ++level)
  {
    glAssert(gl->glTexturePageCommitmentEXT(
        mTexId, level, 0, 0, slice, levelWidth, levelHeight, 1, commit));
    levelWidth = std::max(levelWidth / 2, 1);
    levelHeight = std::max(levelHeight / 2, 1);
  }
}

}  // namespace Graphics
