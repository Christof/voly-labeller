#ifndef SRC_GRAPHICS_TEXTURE_SPACE_DESCRIPTION_H_

#define SRC_GRAPHICS_TEXTURE_SPACE_DESCRIPTION_H_

namespace Graphics
{

struct TextureSpaceDescription
{
  TextureSpaceDescription(int levels, int internalFormat, int width,
                          int height);

  int levels;
  int internalFormat;
  int width;
  int height;
};

bool operator<(const TextureSpaceDescription &left,
               const TextureSpaceDescription &right);

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE_SPACE_DESCRIPTION_H_
