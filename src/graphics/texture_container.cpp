#include "./texture_container.h"
#include <QLoggingCategory>
#include <cassert>
#include <algorithm>
#include <vector>
#include "./gl.h"
#include "./texture2d.h"

namespace Graphics
{

QLoggingCategory tcChan("Graphics.TextureContainer");

TextureContainer::TextureContainer(Gl *gl, bool sparse,
                                   TextureSpaceDescription spaceDescription,
                                   int slices)
  : gl(gl), spaceDescription(spaceDescription), slices(slices)
{
  glAssert(gl->glGenTextures(1, &textureId));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D_ARRAY, textureId));

  if (sparse)
  {
    glAssert(gl->glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_SPARSE_ARB,
                                 GL_TRUE));

    int bestIndex = findBestIndex();
    // This would mean the implementation has no valid sizes for us, or that
    // this format doesn't actually support sparse texture allocation.
    // TODO(all): Need to implement the fallback.
    assert(bestIndex != -1);

    glAssert(gl->glTexParameteri(GL_TEXTURE_2D_ARRAY,
                                 GL_VIRTUAL_PAGE_SIZE_INDEX_ARB, bestIndex));
  }

  // We've set all the necessary parameters, now it's time to create the sparse
  // texture.
  qCDebug(tcChan) << "Container: levels:" << spaceDescription.levels
                  << "width:" << spaceDescription.width
                  << "height:" << spaceDescription.height << "slices:" << slices
                  << "internal format:" << spaceDescription.internalFormat;

  glAssert(gl->glTexStorage3D(GL_TEXTURE_2D_ARRAY, spaceDescription.levels,
                              spaceDescription.internalFormat,
                              spaceDescription.width, spaceDescription.height,
                              slices));

  const uint textureSize =
      spaceDescription.width * spaceDescription.height * slices * 3;
  std::vector<unsigned char> textureData(textureSize, 0);

  glAssert(gl->glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0,
                               spaceDescription.width, spaceDescription.height,
                               slices, GL_RGB, GL_UNSIGNED_BYTE,
                               textureData.data()));

  for (int i = 0; i < slices; ++i)
  {
    freeList.push(i);
  }

  if (sparse)
  {
    handle = gl->getBindlessTexture()->glGetTextureHandleNV(textureId);
    glCheckError();
    assert(handle != 0);

    qCDebug(tcChan) << "Container: textureId:" << textureId
                    << "handle:" << handle;

    glAssert(gl->getBindlessTexture()->glMakeTextureHandleResidentNV(handle));
  }
}

TextureContainer::~TextureContainer()
{
  qCInfo(tcChan) << "Destructor" << spaceDescription.toString().c_str()
                 << "freeList size" << freeList.size() << "slices" << slices;
  if (handle != 0)
  {
    glAssert(
        gl->getBindlessTexture()->glMakeTextureHandleNonResidentNV(handle));
    handle = 0;
  }
  glAssert(gl->glDeleteTextures(1, &textureId));
}

int TextureContainer::hasRoom() const
{
  return freeList.size() > 0;
}

int TextureContainer::virtualAlloc()
{
  qCDebug(tcChan) << "VirtualAlloc";
  int returnValue = freeList.front();
  freeList.pop();

  return returnValue;
}

void TextureContainer::virtualFree(int slice)
{
  qCDebug(tcChan) << "Virtual free for slice" << slice;
  freeList.push(slice);
}

void TextureContainer::commit(Texture2d *texture)
{
  assert(texture->getTextureContainer() == this);

  changeCommitment(texture->getSliceCount(), GL_TRUE);
}

void TextureContainer::free(Texture2d *texture)
{
  assert(texture->getTextureContainer() == this);
  changeCommitment(texture->getSliceCount(), GL_FALSE);
}

void TextureContainer::compressedTexSubImage3d(int level, int xOffset,
                                               int yOffset, int zOffset,
                                               int width, int height, int depth,
                                               int format, int imageSize,
                                               const void *data)
{
  glAssert(gl->glBindTexture(GL_TEXTURE_2D_ARRAY, textureId));
  glAssert(gl->glCompressedTexSubImage3D(GL_TEXTURE_2D_ARRAY, level, xOffset,
                                         yOffset, zOffset, width, height, depth,
                                         format, imageSize, data));
}

void TextureContainer::texSubImage3d(int level, int xOffset, int yOffset,
                                     int zOffset, int width, int height,
                                     int depth, int format, int type,
                                     const void *data)
{
  glAssert(gl->glBindTexture(GL_TEXTURE_2D_ARRAY, textureId));

  qCDebug(tcChan) << "In texSubImage3d level:" << level << "xOffset:" << xOffset
                  << "yOffset:" << yOffset << "zOffset:" << zOffset
                  << "width:" << width << "height:" << height
                  << "depth:" << depth << "format:" << format
                  << "type:" << type;

  if (data == nullptr)
  {
    float clearValue[4] = { 0, 0, 0, 0 };
    gl->glClearTexSubImage(textureId, level, xOffset, yOffset, zOffset, width,
                           height, depth, format, type, clearValue);
  }
  else
  {
    glAssert(gl->glTexSubImage3D(GL_TEXTURE_2D_ARRAY, level, xOffset, yOffset,
                                 zOffset, width, height, depth, format, type,
                                 data));
  }
}

unsigned long int TextureContainer::getHandle() const
{
  return handle;
}

int TextureContainer::getWidth() const
{
  return spaceDescription.width;
}

int TextureContainer::getHeight() const
{
  return spaceDescription.height;
}

int TextureContainer::findBestIndex()
{
  // TODO(all): This could be done once per internal format. For now, just do it
  // every time.

  int bestIndex = -1;
  int bestXSize = 0;
  int bestYSize = 0;

  int indexCount = 0;
  glAssert(gl->glGetInternalformativ(
      GL_TEXTURE_2D_ARRAY, spaceDescription.internalFormat,
      GL_NUM_VIRTUAL_PAGE_SIZES_ARB, 1, &indexCount));

  qCDebug(tcChan) << "Container: indexCount:" << indexCount
                  << "width:" << spaceDescription.width
                  << "height:" << spaceDescription.height
                  << "internal Format:" << spaceDescription.internalFormat;

  for (int i = 0; i < indexCount; ++i)
  {
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_VIRTUAL_PAGE_SIZE_INDEX_ARB, i);
    int xSize = 0;
    glAssert(gl->glGetInternalformativ(GL_TEXTURE_2D_ARRAY,
                                       spaceDescription.internalFormat,
                                       GL_VIRTUAL_PAGE_SIZE_X_ARB, 1, &xSize));
    int ySize = 0;
    glAssert(gl->glGetInternalformativ(GL_TEXTURE_2D_ARRAY,
                                       spaceDescription.internalFormat,
                                       GL_VIRTUAL_PAGE_SIZE_Y_ARB, 1, &ySize));
    int zSize = 0;
    glAssert(gl->glGetInternalformativ(GL_TEXTURE_2D_ARRAY,
                                       spaceDescription.internalFormat,
                                       GL_VIRTUAL_PAGE_SIZE_Z_ARB, 1, &zSize));

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

  return bestIndex;
}

void TextureContainer::changeCommitment(int slice, bool commit)
{
  int levelWidth = spaceDescription.width;
  int levelHeight = spaceDescription.height;

  for (int level = 0; level < spaceDescription.levels; ++level)
  {
    glAssert(gl->glTexturePageCommitmentEXT(
        textureId, level, 0, 0, slice, levelWidth, levelHeight, 1, commit));
    levelWidth = std::max(levelWidth / 2, 1);
    levelHeight = std::max(levelHeight / 2, 1);
  }
}

}  // namespace Graphics
