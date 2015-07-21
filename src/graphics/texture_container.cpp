#include "./texture_container.h"
#include <cassert>
#include <iostream>
#include "./texture2d.h"

namespace Graphics
{

TextureContainer::TextureContainer(Gl *gl, bool sparse, GLsizei levels,
                                   GLenum internalformat, GLsizei width,
                                   GLsizei height, GLsizei slices)
  : gl(gl), width(width), height(height), levels(levels), slices(slices)
{
  glAssert(gl->glGenTextures(1, &textureId));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D_ARRAY, textureId));

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

    glAssert(gl->glTexParameteri(GL_TEXTURE_2D_ARRAY,
                                 GL_VIRTUAL_PAGE_SIZE_INDEX_ARB, bestIndex));
  }

  // We've set all the necessary parameters, now it's time to create the sparse
  // texture.
  std::cout << "Container: levels:" << levels << " width: " << width
            << " height: " << height << " slices: " << slices
            << " internal format: " << internalformat << std::endl;

  glAssert(gl->glTexStorage3D(GL_TEXTURE_2D_ARRAY, levels, internalformat,
                              width, height, slices));

  const uint tsize = width * height * slices * 3;
  unsigned char *tdata = new unsigned char[tsize];
  for (uint i = 0; i < tsize; i++)
    tdata[i] = 0;

  glAssert(gl->glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, width, height,
                               slices, GL_RGB, GL_UNSIGNED_BYTE, tdata));

  for (GLsizei i = 0; i < slices; ++i)
  {
    freeList.push(i);
  }

  if (sparse)
  {
    handle = gl->getBindlessTexture()->glGetTextureHandleNV(textureId);
    glCheckError();
    std::cout << "Container: textureId: " << textureId << " handle: " << handle
              << std::endl;
    assert(handle != 0);
    glAssert(gl->getBindlessTexture()->glMakeTextureHandleResidentNV(handle));
  }
}

TextureContainer::~TextureContainer()
{
  // If this fires, it means there was a texture leaked somewhere.
  assert(freeList.size() == static_cast<size_t>(slices));

  if (handle != 0)
  {
    glAssert(
        gl->getBindlessTexture()->glMakeTextureHandleNonResidentNV(handle));
    handle = 0;
  }
  glAssert(gl->glDeleteTextures(1, &textureId));
}

GLsizei TextureContainer::hasRoom() const
{
  return freeList.size() > 0;
}

GLsizei TextureContainer::virtualAlloc()
{
  GLsizei returnValue = freeList.front();
  freeList.pop();

  return returnValue;
}

void TextureContainer::virtualFree(GLsizei slice)
{
  freeList.push(slice);
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

void TextureContainer::compressedTexSubImage3d(GLint level, GLint xoffset,
                                               GLint yoffset, GLint zoffset,
                                               GLsizei width, GLsizei height,
                                               GLsizei depth, GLenum format,
                                               GLsizei imageSize,
                                               const GLvoid *data)
{
  glAssert(gl->glBindTexture(GL_TEXTURE_2D_ARRAY, textureId));
  glAssert(gl->glCompressedTexSubImage3D(GL_TEXTURE_2D_ARRAY, level, xoffset,
                                         yoffset, zoffset, width, height, depth,
                                         format, imageSize, data));
}

void TextureContainer::texSubImage3d(GLint level, GLint xoffset, GLint yoffset,
                                     GLint zoffset, GLsizei width,
                                     GLsizei height, GLsizei depth,
                                     GLenum format, GLenum type,
                                     const GLvoid *data)
{
  glAssert(gl->glBindTexture(GL_TEXTURE_2D_ARRAY, textureId));

  std::cout << "level: " << level << " xoffset: " << xoffset
            << " yoffset: " << yoffset << " zoffset: " << zoffset
            << " width: " << width << " height: " << height
            << " depth: " << depth << " format: " << format << " type: " << type
            << std::endl;

  glAssert(gl->glTexSubImage3D(GL_TEXTURE_2D_ARRAY, level, xoffset, yoffset,
                               zoffset, width, height, depth, format, type,
                               data));
}

GLuint64 TextureContainer::getHandle() const
{
  return handle;
}

GLsizei TextureContainer::getWidth() const
{
  return width;
}

GLsizei TextureContainer::getHeight() const
{
  return height;
}

void TextureContainer::changeCommitment(GLsizei slice, GLboolean commit)
{
  GLsizei levelWidth = width;
  GLsizei levelHeight = height;

  for (int level = 0; level < levels; ++level)
  {
    glAssert(gl->glTexturePageCommitmentEXT(
        textureId, level, 0, 0, slice, levelWidth, levelHeight, 1, commit));
    levelWidth = std::max(levelWidth / 2, 1);
    levelHeight = std::max(levelHeight / 2, 1);
  }
}

}  // namespace Graphics
