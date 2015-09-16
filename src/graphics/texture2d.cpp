#include "./texture2d.h"
#include "./texture_container.h"

namespace Graphics
{

Texture2d::Texture2d(TextureContainer *container, int sliceCount, int width,
                     int height)
  : container(container), width(width), height(height), sliceCount(sliceCount)
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

int Texture2d::getSliceCount() const
{
  return sliceCount;
}

TextureAddress Texture2d::address() const
{
  return { container->getHandle(),
           static_cast<float>(sliceCount),
           0,
           { static_cast<float>(width) / container->getWidth(),
             static_cast<float>(height) / container->getHeight() } };
}

void Texture2d::commit()
{
  container->commit(this);
}

void Texture2d::free()
{
  container->free(this);
}

void Texture2d::compressedTexSubImage2D(int level, int xOffset, int yOffset,
                                        int width, int height, int format,
                                        int imageSize, const void *data)
{
  container->compressedTexSubImage3d(level, xOffset, yOffset, sliceCount, width,
                                     height, 1, format, imageSize, data);
}

void Texture2d::texSubImage2D(int level, int xOffset, int yOffset, int width,
                              int height, int format, int type,
                              const void *data)
{
  container->texSubImage3d(level, xOffset, yOffset, sliceCount, width, height,
                           1, format, type, data);
}

}  // namespace Graphics
