#include "./texture_space_description.h"
#include <string>

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

int computeNextPowerOfTwo(int value)
{
  int powerOfTwo = 1;
  while (powerOfTwo < value)
    powerOfTwo <<= 1;

  return powerOfTwo;
}

void TextureSpaceDescription::growToValidSize(int minX, int minY)
{
  if (width < minX)
    width = minX;
  if (height < minY)
    height = minY;

  width = computeNextPowerOfTwo(width);
  height = computeNextPowerOfTwo(height);
}

std::string TextureSpaceDescription::toString() const
{
  return "Levels: " + std::to_string(levels) + " width: " +
         std::to_string(width) + " height: " + std::to_string(height);
}

}  // namespace Graphics
