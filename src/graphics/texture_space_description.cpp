#include "./texture_space_description.h"

namespace Graphics
{

TextureSpaceDescription::TextureSpaceDescription(int levels, int internalFormat,
                                                 int width, int height)
  : levels(levels), internalFormat(internalFormat), width(width), height(height)
{
}

bool operator<(const TextureSpaceDescription &left,
               const TextureSpaceDescription &right)
{
  if (left.width < right.width)
    return true;
  if (left.width > right.width)
    return false;

  if (left.height < right.height)
    return true;
  if (left.height > right.height)
    return false;

  if (left.levels < right.levels)
    return true;
  if (left.levels > right.levels)
    return false;

  if (left.internalFormat < right.internalFormat)
    return true;
  if (left.internalFormat > right.internalFormat)
    return false;

  return false;
}

}  // namespace Graphics
