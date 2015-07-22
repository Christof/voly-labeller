#ifndef SRC_GRAPHICS_TEXTURE_CONTAINER_H_

#define SRC_GRAPHICS_TEXTURE_CONTAINER_H_

#include <queue>
#include "./texture_space_description.h"

namespace Graphics
{

class Texture2d;
class Gl;

class TextureContainer
{
 public:
  TextureContainer(Gl *gl, bool sparse,
                   TextureSpaceDescription spaceDescription, int slices);
  ~TextureContainer();
  int hasRoom() const;
  int virtualAlloc();
  void virtualFree(int slice);

  void commit(Texture2d *texture);
  void free(Texture2d *texture);

  void compressedTexSubImage3d(int level, int xOffset, int yOffset, int zOffset,
                               int width, int height, int depth, int format,
                               int imageSize, const void *data);
  void texSubImage3d(int level, int xOffset, int yOffset, int zOffset,
                     int width, int height, int depth, int format, int type,
                     const void *data);

  unsigned long int getHandle() const;
  int getWidth() const;
  int getHeight() const;

 private:
  Gl *gl;
  unsigned long int handle = 0;
  unsigned int textureId;
  std::queue<int> freeList;

  const TextureSpaceDescription spaceDescription;
  const int slices;

  void changeCommitment(int slice, bool commit);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE_CONTAINER_H_
