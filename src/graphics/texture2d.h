#ifndef SRC_GRAPHICS_TEXTURE2D_H_

#define SRC_GRAPHICS_TEXTURE2D_H_

#include "./texture_address.h"

namespace Graphics
{

class TextureContainer;

/**
 * \brief 2d bindless texture which is stored in a TextureContainer
 *
 */
class Texture2d
{
 public:
  Texture2d(TextureContainer *container, int sliceIndex, int width, int height);
  ~Texture2d();
  void commit();
  void free();

  void compressedTexSubImage2D(int level, int xOffset, int yOffset, int width,
                               int height, int format, int imageSize,
                               const void *data);
  void texSubImage2D(int level, int xOffset, int yOffset, int width, int height,
                     int format, int type, const void *data);

  const TextureContainer *getTextureContainer() const;

  int getSliceIndex() const;

  TextureAddress address() const;

 private:
  TextureContainer *container;

  int width;
  int height;
  int sliceIndex;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE2D_H_
