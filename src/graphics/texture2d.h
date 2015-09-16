#ifndef SRC_GRAPHICS_TEXTURE2D_H_

#define SRC_GRAPHICS_TEXTURE2D_H_

#include "./texture_address.h"

namespace Graphics
{

class TextureContainer;

/**
 * \brief
 *
 *
 */
class Texture2d
{
 public:
  Texture2d(TextureContainer *container, int sliceCount, int width, int height);
  ~Texture2d();
  void commit();
  void free();

  void compressedTexSubImage2D(int level, int xOffset, int yOffset, int width,
                               int height, int format, int imageSize,
                               const void *data);
  void texSubImage2D(int level, int xOffset, int yOffset, int width, int height,
                     int format, int type, const void *data);

  const TextureContainer *getTextureContainer() const;

  int getSliceCount() const;

  TextureAddress address() const;

 private:
  TextureContainer *container;

  int width;
  int height;
  int sliceCount;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE2D_H_
