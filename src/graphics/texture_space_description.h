#ifndef SRC_GRAPHICS_TEXTURE_SPACE_DESCRIPTION_H_

#define SRC_GRAPHICS_TEXTURE_SPACE_DESCRIPTION_H_

#include <string>

namespace Graphics
{

/**
 * \brief Describes space requirements for a texture
 *
 * This consists of the #width and #height, the #levels as well as the
 * \link #internalFormat internal format\endlink.
 */
struct TextureSpaceDescription
{
  TextureSpaceDescription(int levels, int internalFormat, int width,
                          int height);

  int levels;
  int internalFormat;
  int width;
  int height;

  /**
   * \brief Grows the size to valid size
   *
   * This means it must be at least the virtual page size given by \p minX and
   * \p minY and it must be a power of 2 in each dimension.
   */
  void growToValidSize(int minX, int minY);

  std::string toString() const;
};

bool operator<(const TextureSpaceDescription &left,
               const TextureSpaceDescription &right);

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE_SPACE_DESCRIPTION_H_
